"""
backend/services/query_understanding.py
─────────────────────────────────────────────────────────────
Query Understanding, Intent Classification, and Decomposition Service (Phase 11).

Key responsibilities:
  1. QueryNormalizer: Clean excess whitespace and polite conversational preambles
     while strictly preserving technical terms, punctuation, and casing.
  2. IntentClassifier: Categorize query into FACTUAL, COMPARATIVE, MULTI_PART,
     SUMMARIZATION, or EXPLORATORY intent using deterministic linguistic patterns.
  3. QueryDecomposer: Detect genuine multi-requirement queries without splitting
     on false-positive noun conjunctions ("research and development", "oil and gas").
  4. Associate every generated SubQuery with the immutable original parent query.
─────────────────────────────────────────────────────────────
"""

import re
from typing import List, Optional, Set, Tuple

from backend.schemas.query_plan import QueryAnalysis, QueryIntent, SubQuery


class QueryNormalizer:
    """
    Cleans and normalizes query text while preserving technical symbols,
    domain identifiers, and original phrasing.
    """

    # Conversational preambles to strip from the beginning of queries
    _PREAMBLE_PATTERNS = [
        re.compile(r"^(?:could|can)\s+you\s+(?:please\s+)?(?:tell\s+me|explain|provide|show\s+me|give\s+me|describe)\s+", re.IGNORECASE),
        re.compile(r"^(?:please\s+)?(?:tell\s+me\s+about|explain|describe|show\s+me|give\s+me\s+information\s+on)\s+", re.IGNORECASE),
        re.compile(r"^(?:i\s+(?:would\s+like|want)\s+to\s+know|i'm\s+wondering)\s+(?:about\s+|whether\s+|if\s+)?", re.IGNORECASE),
        re.compile(r"^(?:what\s+can\s+you\s+tell\s+me\s+about)\s+", re.IGNORECASE),
        re.compile(r"^(?:kindly\s+(?:explain|tell\s+me|clarify))\s+", re.IGNORECASE),
    ]

    # Redundant consecutive horizontal whitespace
    _WHITESPACE_RE = re.compile(r"[^\S\r\n]+")

    @classmethod
    def normalize(cls, query: str) -> str:
        """
        Normalize query string:
          1. Collapse horizontal spaces and trim leading/trailing whitespace.
          2. Strip conversational preambles if a substantive question remains.
          3. Clean extraneous repeated punctuation (e.g. '??' -> '?').
        """
        if not query or not query.strip():
            return ""

        cleaned = cls._WHITESPACE_RE.sub(" ", query).strip()

        # Try stripping conversational preambles
        for pattern in cls._PREAMBLE_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                remainder = cleaned[match.end():].strip()
                # Ensure we don't reduce the query to nothing
                if len(remainder) >= 3:
                    cleaned = remainder
                    break

        # Collapse multiple question marks or exclamation marks
        cleaned = re.sub(r"\?{2,}", "?", cleaned)
        cleaned = re.sub(r"!{2,}", "!", cleaned)

        return cleaned.strip()


class IntentClassifier:
    """
    Deterministically classifies query intent into standard categories.
    """

    _COMPARATIVE_RE = re.compile(
        r"\b(?:compare|comparison|versus|vs\.?|diff(?:erence)?\s+between|pros\s+and\s+cons\s+of|trade-?offs?\s+between)\b",
        re.IGNORECASE,
    )
    _SUMMARIZATION_RE = re.compile(
        r"\b(?:summarize|summary|overview|briefly\s+outline|synopsis|main\s+points|key\s+takeaways|recap|abstract)\b",
        re.IGNORECASE,
    )
    _EXPLORATORY_RE = re.compile(
        r"\b(?:how\s+does|how\s+do|why\s+is|why\s+are|why\s+does|how\s+can|explain\s+how|mechanism\s+of|architecture\s+of)\b",
        re.IGNORECASE,
    )
    _MULTI_PART_RE = re.compile(
        r"(?:\?[\s\w]+\?|\band\s+(?:what|how|why|who|when|where|which|also|additionally)\b|;\s*(?:what|how|why|who|when))",
        re.IGNORECASE,
    )

    @classmethod
    def classify(cls, query: str) -> QueryIntent:
        """
        Classify query intent based on linguistic anchors.
        Evaluation priority:
          1. Comparative (highest specificity)
          2. Summarization
          3. Multi-part questions
          4. Exploratory / How / Why
          5. Factual (default for direct entity queries)
        """
        q_lower = query.lower()

        if cls._COMPARATIVE_RE.search(q_lower):
            return QueryIntent.COMPARATIVE
        if cls._SUMMARIZATION_RE.search(q_lower):
            return QueryIntent.SUMMARIZATION
        if cls._MULTI_PART_RE.search(q_lower):
            return QueryIntent.MULTI_PART
        if cls._EXPLORATORY_RE.search(q_lower):
            return QueryIntent.EXPLORATORY

        return QueryIntent.FACTUAL

    @classmethod
    def classify_college_intent(cls, query: str) -> Optional[QueryIntent]:
        """
        Classifies domain-specific college institutional intents (Phase 23).
        """
        q = query.lower()
        if re.search(r"\b(?:where\s+is|cabin|room\s+no|block|floor|locate|location\s+of|hod\s+cabin)\b", q):
            return QueryIntent.CAMPUS_LOCATION
        if re.search(r"\b(?:placement|placements|package|placed|eligibility\s+conditions?|recruitment|recruiter|campus\s+drive)\b", q):
            return QueryIntent.PLACEMENT_POLICY
        if re.search(r"\b(?:internship|internships|intern|summer\s+intern|industrial\s+training)\b", q):
            return QueryIntent.INTERNSHIP_POLICY
        if re.search(r"\b(?:attendance|absent|shortage|condonation|minimum\s+attendance)\b", q):
            return QueryIntent.ATTENDANCE_POLICY
        if re.search(r"\b(?:exam|examination|grading|gpa|cgpa|backlog|supplementary|evaluation|revaluation|hall\s+ticket)\b", q):
            return QueryIntent.EXAMINATION_POLICY
        if re.search(r"\b(?:admission|admissions|eligibility\s+requirements?|intake|documents\s+needed|cutoff)\b", q):
            return QueryIntent.ADMISSION_QUERY
        if re.search(r"\b(?:course|courses|program|programs|curriculum|syllabus|branch|branches|b\.?tech|m\.?tech|mba)\b", q):
            return QueryIntent.COURSE_QUERY
        if re.search(r"\b(?:scholarship|scholarships|financial\s+aid|concession|freeship)\b", q):
            return QueryIntent.SCHOLARSHIP_QUERY
        if re.search(r"\b(?:fee|fees|tuition|installment|cost|dues)\b", q):
            return QueryIntent.FEE_QUERY
        if re.search(r"\b(?:leave\s+procedure|faculty\s+leave|casual\s+leave|academic\s+leave|hr\s+polic|faculty\s+duties)\b", q):
            return QueryIntent.FACULTY_POLICY
        if re.search(r"\b(?:facility|facilities|hostel|mess|library|lab|laborator|canteen|sports|gym|wifi)\b", q):
            return QueryIntent.FACILITY_INFORMATION
        if re.search(r"\b(?:contact|whom\s+should\s+i|phone|email|office\s+hours|helpline|reach)\b", q):
            return QueryIntent.CONTACT_QUERY
        if re.search(r"\b(?:department|cse|it|ece|eee|mech|civil|ai&ds|hod)\b", q):
            return QueryIntent.DEPARTMENT_INFORMATION
        if re.search(r"\b(?:administrative|procedure|governance|policies\s+uploaded|missing\s+information|updating)\b", q):
            return QueryIntent.ADMINISTRATIVE_QUERY
        return None

    @classmethod
    def extract_target_categories(cls, query: str, college_intent: Optional[QueryIntent]) -> List[str]:
        """Map query intent and keywords to institutional categories."""
        cats = []
        if college_intent == QueryIntent.CAMPUS_LOCATION:
            cats.extend(["Campus Directory", "Departments"])
        elif college_intent == QueryIntent.PLACEMENT_POLICY:
            cats.extend(["Placement Policies", "Departments"])
        elif college_intent == QueryIntent.INTERNSHIP_POLICY:
            cats.extend(["Internship Guidelines"])
        elif college_intent == QueryIntent.ATTENDANCE_POLICY:
            cats.extend(["Attendance Regulations"])
        elif college_intent == QueryIntent.EXAMINATION_POLICY:
            cats.extend(["Examination Regulations", "Academic Calendar"])
        elif college_intent == QueryIntent.ADMISSION_QUERY:
            cats.extend(["Admissions", "Courses and Programs"])
        elif college_intent == QueryIntent.COURSE_QUERY:
            cats.extend(["Courses and Programs", "Departments"])
        elif college_intent == QueryIntent.SCHOLARSHIP_QUERY:
            cats.extend(["Scholarships", "Fee Structures"])
        elif college_intent == QueryIntent.FEE_QUERY:
            cats.extend(["Fee Structures"])
        elif college_intent == QueryIntent.FACULTY_POLICY:
            cats.extend(["Faculty HR Policies", "Administrative Procedures"])
        elif college_intent == QueryIntent.FACILITY_INFORMATION:
            cats.extend(["Laboratories and Facilities", "Hostel Information", "Library Information"])
        elif college_intent == QueryIntent.CONTACT_QUERY:
            cats.extend(["Campus Directory", "Student Services"])
        elif college_intent == QueryIntent.ADMINISTRATIVE_QUERY:
            cats.extend(["Administrative Procedures", "General College Information"])
        return list(dict.fromkeys(cats))

    @classmethod
    def extract_target_departments(cls, query: str) -> List[str]:
        """Detect department mentions in the query."""
        q = query.lower()
        depts = []
        if re.search(r"\b(?:cse|computer\s+science)\b", q):
            depts.append("Computer Science and Engineering (CSE)")
        if re.search(r"\b(?:ece|electronics)\b", q):
            depts.append("Electronics and Communication (ECE)")
        if re.search(r"\b(?:it|information\s+technology)\b", q):
            depts.append("Information Technology (IT)")
        if re.search(r"\b(?:mech|mechanical)\b", q):
            depts.append("Mechanical Engineering (ME)")
        if re.search(r"\b(?:civil)\b", q):
            depts.append("Civil Engineering (CE)")
        if re.search(r"\b(?:ai|data\s+science|ai&ds)\b", q):
            depts.append("Artificial Intelligence and Data Science (AI&DS)")
        if re.search(r"\b(?:placement|recruit)\b", q):
            depts.append("Placement Cell")
        if re.search(r"\b(?:admission)\b", q):
            depts.append("Admissions Office")
        if re.search(r"\b(?:exam|examination)\b", q):
            depts.append("Examination Cell")
        return list(dict.fromkeys(depts))


class SharedConstraintExtractor:
    """
    Detects and preserves shared contextual constraints across decomposed subqueries,
    including financial quarters, dates, years, locations, conditions, and named entities.
    """

    _TEMPORAL_PHRASE_RE = re.compile(
        r"\b(?:in|for|during|of|throughout)\s+(?:Q[1-4](?:\s+\d{4})?|FY\s*\d{2,4}|fiscal\s+year\s+\d{4}|\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{1,2})?(?:,?\s+\d{4})?)\b",
        re.IGNORECASE,
    )
    _BARE_QUARTER_RE = re.compile(r"\bQ[1-4]\s+\d{4}\b", re.IGNORECASE)
    _BARE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

    _LOCATION_PHRASE_RE = re.compile(
        r"\b(?:in|across|throughout)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b"
    )
    _LOCATION_EXCLUSIONS = {
        "detail", "general", "summary", "brief", "short", "total", "addition",
        "contrast", "comparison", "fact", "practice", "python", "java", "c++",
        "sql", "rust", "q1", "q2", "q3", "q4", "each", "both", "all", "terms",
        "what", "how", "why", "who", "when", "where", "which", "january",
        "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december"
    }

    _CONDITION_PHRASE_RE = re.compile(
        r"\b(?:under|pursuant\s+to|according\s+to|per)\s+([A-Z0-9]+(?:\s+[A-Z0-9]+)*)\b",
        re.IGNORECASE,
    )

    _NAMED_ENTITY_PREP_RE = re.compile(
        r"\b(?:at|for|of)\s+([A-Z][a-zA-Z0-9]+(?:\s+(?:Corp|Corporation|Inc|LLC|Technologies|Labs|Group|Company|AI|Systems))?)\b"
    )
    _NAMED_ENTITY_BARE_RE = re.compile(
        r"\b([A-Z][a-zA-Z0-9]+(?:\s+(?:Corp|Corporation|Inc|LLC|Technologies|Labs|Group|Company|AI|Systems)))\b"
    )
    _PRONOUN_RE = re.compile(r"\b(?:its|their)\b", re.IGNORECASE)

    @classmethod
    def extract_temporal(cls, text: str) -> Optional[str]:
        m = cls._TEMPORAL_PHRASE_RE.search(text)
        if m:
            phrase = m.group(0).strip()
            return phrase[0].lower() + phrase[1:]
        m_q = cls._BARE_QUARTER_RE.search(text)
        if m_q:
            return f"in {m_q.group(0).strip()}"
        return None

    @classmethod
    def extract_location(cls, text: str) -> Optional[str]:
        for m in cls._LOCATION_PHRASE_RE.finditer(text):
            candidate = m.group(1).strip()
            if candidate.lower() not in cls._LOCATION_EXCLUSIONS:
                if not re.match(r"^(?:Q[1-4]|\d{4})$", candidate, re.IGNORECASE):
                    phrase = m.group(0).strip()
                    return phrase[0].lower() + phrase[1:]
        return None

    @classmethod
    def extract_condition(cls, text: str) -> Optional[str]:
        m = cls._CONDITION_PHRASE_RE.search(text)
        if m:
            phrase = m.group(0).strip()
            return phrase[0].lower() + phrase[1:]
        return None

    @classmethod
    def extract_entity(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        m_prep = cls._NAMED_ENTITY_PREP_RE.search(text)
        if m_prep:
            ent = m_prep.group(1).strip()
            if ent.lower() not in cls._LOCATION_EXCLUSIONS:
                phrase = m_prep.group(0).strip()
                return ent, phrase[0].lower() + phrase[1:]

        m_bare = cls._NAMED_ENTITY_BARE_RE.search(text)
        if m_bare:
            ent = m_bare.group(1).strip()
            return ent, f"for {ent}"

        return None, None

    @classmethod
    def propagate_constraints(cls, clauses: List[str]) -> List[str]:
        """
        Propagate identified temporal, location, condition, and entity constraints
        to any clauses that lack them.
        """
        shared_temporal: Optional[str] = None
        shared_location: Optional[str] = None
        shared_condition: Optional[str] = None
        shared_entity_name: Optional[str] = None
        shared_entity_prep: Optional[str] = None

        for c in clauses:
            if not shared_temporal:
                shared_temporal = cls.extract_temporal(c)
            if not shared_location:
                shared_location = cls.extract_location(c)
            if not shared_condition:
                shared_condition = cls.extract_condition(c)
            if not shared_entity_name:
                ent_name, ent_prep = cls.extract_entity(c)
                if ent_name:
                    shared_entity_name = ent_name
                    shared_entity_prep = ent_prep

        enriched: List[str] = []
        for c in clauses:
            clean = c.strip(" ?,.")

            # 1. Named Entity / Pronoun substitution
            if shared_entity_name:
                if cls._PRONOUN_RE.search(clean):
                    clean = cls._PRONOUN_RE.sub(f"{shared_entity_name}'s", clean)
                elif shared_entity_name.lower() not in clean.lower():
                    if not cls.extract_entity(clean)[0]:
                        prep = shared_entity_prep or f"for {shared_entity_name}"
                        clean = f"{clean} {prep}"

            # 2. Regulatory Condition
            if shared_condition:
                cond_key = shared_condition.split()[-1]
                if cond_key.lower() not in clean.lower() and not cls.extract_condition(clean):
                    clean = f"{clean} {shared_condition}"

            # 3. Location
            if shared_location:
                loc_key = shared_location.split()[-1]
                if loc_key.lower() not in clean.lower() and not cls.extract_location(clean):
                    clean = f"{clean} {shared_location}"

            # 4. Temporal (Quarters, Years, Dates)
            if shared_temporal:
                has_temp = cls.extract_temporal(clean) is not None
                temp_keys = [w for w in shared_temporal.split() if re.match(r"^(?:Q[1-4]|\d{4})$", w, re.IGNORECASE)]
                has_keys = any(k.lower() in clean.lower() for k in temp_keys)
                if not has_temp and not has_keys:
                    clean = f"{clean} {shared_temporal}"

            clean = re.sub(r"\s+", " ", clean).strip()
            if clean:
                clean = clean[0].upper() + clean[1:]
                if not clean.endswith("?"):
                    clean += "?"
            enriched.append(clean)

        return enriched


class QueryDecomposer:
    """
    Decomposes multi-faceted or compound queries into self-contained SubQuery objects.
    Protects established multi-word entities and compound phrases from false splitting.
    Preserves shared constraints (dates, quarters, locations, entities, conditions)
    and comparative objectives.
    """

    # Common multi-word phrases and idioms containing 'and' that must NOT be split
    _PROTECTED_AND_PHRASES: Set[str] = {
        "research and development",
        "oil and gas",
        "apples and oranges",
        "mergers and acquisitions",
        "hardware and software",
        "profit and loss",
        "sales and marketing",
        "tips and tricks",
        "pros and cons",
        "terms and conditions",
        "science and technology",
        "laws and regulations",
        "peace and security",
        "command and control",
        "checks and balances",
        "supply and demand",
        "search and rescue",
        "plug and play",
    }

    # Interrogative conjunction boundary (e.g. "... and what was ...", "... and how do ...")
    _INTERROGATIVE_CONJUNCTION_RE = re.compile(
        r",?\s*\band\s+(?=(?:what|how|why|who|when|where|which|describe|explain|detail|outline)\b)",
        re.IGNORECASE,
    )

    # Question mark sequence boundary ("What is X? What is Y?")
    _QUESTION_SPLIT_RE = re.compile(r"\?\s+(?=[A-Z0-9])")

    # Semicolon boundary ("What is X; also what is Y")
    _SEMICOLON_RE = re.compile(r";\s*")

    # Comparative pattern: "Compare X and Y" or "Difference between X and Y"
    _COMPARE_AND_RE = re.compile(
        r"(?:compare|difference\s+between|versus|vs\.?)\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+)",
        re.IGNORECASE,
    )

    @classmethod
    def decompose(
        cls,
        original_query: str,
        normalized_query: str,
        intent: QueryIntent,
    ) -> Tuple[bool, List[str], List[SubQuery], Optional[str]]:
        """
        Evaluate complexity and derive subqueries while preserving shared constraints
        and comparative objectives.

        Returns:
            Tuple of:
              - is_complex (bool)
              - complexity_factors (List[str])
              - subqueries (List[SubQuery])
              - comparison_objective (Optional[str])
        """
        # 1. Check for comparative queries
        if intent == QueryIntent.COMPARATIVE or cls._COMPARE_AND_RE.search(normalized_query):
            compare_match = cls._COMPARE_AND_RE.search(normalized_query)
            if compare_match:
                entity_a = compare_match.group(1).strip(" ?,.")
                entity_b = compare_match.group(2).strip(" ?,.")
                if entity_a and entity_b and entity_a.lower() != entity_b.lower():
                    comparison_obj = f"Compare {entity_a} and {entity_b}"
                    sq_a = SubQuery(
                        parent_query=original_query,
                        text=f"Overview, features, and characteristics of {entity_a}",
                        intent=QueryIntent.FACTUAL,
                        order=0,
                        description=f"Information requirement for entity A: {entity_a}",
                    )
                    sq_b = SubQuery(
                        parent_query=original_query,
                        text=f"Overview, features, and characteristics of {entity_b}",
                        intent=QueryIntent.FACTUAL,
                        order=1,
                        description=f"Information requirement for entity B: {entity_b}",
                    )
                    return True, ["comparative_entity_split"], [sq_a, sq_b], comparison_obj
            else:
                comparison_obj = normalized_query if intent == QueryIntent.COMPARATIVE else None
        else:
            comparison_obj = None

        # 2. Check for multiple questions ("What is X? What is Y?")
        q_parts = cls._QUESTION_SPLIT_RE.split(normalized_query)
        if len(q_parts) > 1:
            enriched_parts = SharedConstraintExtractor.propagate_constraints(q_parts)
            if len(enriched_parts) > 1:
                subqueries = [
                    SubQuery(
                        parent_query=original_query,
                        text=part,
                        intent=IntentClassifier.classify(part),
                        order=idx,
                        description=f"Sequential question facet {idx + 1}",
                    )
                    for idx, part in enumerate(enriched_parts)
                ]
                return True, ["multiple_question_sentences"], subqueries, comparison_obj

        # 3. Check for semicolon clause separation
        semi_parts = cls._SEMICOLON_RE.split(normalized_query)
        if len(semi_parts) > 1:
            valid_semi = [p.strip() for p in semi_parts if len(p.strip()) > 5]
            if len(valid_semi) > 1:
                enriched_parts = SharedConstraintExtractor.propagate_constraints(valid_semi)
                subqueries = [
                    SubQuery(
                        parent_query=original_query,
                        text=part,
                        intent=IntentClassifier.classify(part),
                        order=idx,
                        description=f"Semicolon-separated subquery {idx + 1}",
                    )
                    for idx, part in enumerate(enriched_parts)
                ]
                return True, ["semicolon_clauses"], subqueries, comparison_obj

        # 4. Check for interrogative conjunctions ("..., and what is...")
        conj_parts = cls._INTERROGATIVE_CONJUNCTION_RE.split(normalized_query)
        if len(conj_parts) > 1:
            valid_parts = [p.strip(" ?,.") for p in conj_parts if len(p.strip(" ?,.")) > 6]
            if len(valid_parts) > 1:
                enriched_parts = SharedConstraintExtractor.propagate_constraints(valid_parts)
                subqueries = [
                    SubQuery(
                        parent_query=original_query,
                        text=part,
                        intent=IntentClassifier.classify(part),
                        order=idx,
                        description=f"Conjunctive subquery {idx + 1}",
                    )
                    for idx, part in enumerate(enriched_parts)
                ]
                return True, ["interrogative_conjunction"], subqueries, comparison_obj

        # 5. Fallback: Simple or single-facet query (no decomposition)
        single_subquery = SubQuery(
            parent_query=original_query,
            text=normalized_query if normalized_query.endswith("?") or intent == QueryIntent.SUMMARIZATION else f"{normalized_query}?",
            intent=intent,
            order=0,
            description="Single direct information requirement.",
        )
        comp_obj = normalized_query if intent == QueryIntent.COMPARATIVE else comparison_obj
        return False, [], [single_subquery], comp_obj


class QueryUnderstandingService:
    """
    Unified service orchestrating query normalization, intent classification,
    complexity detection, and subquery decomposition.
    """

    def __init__(self):
        self.normalizer = QueryNormalizer()
        self.classifier = IntentClassifier()
        self.decomposer = QueryDecomposer()

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze an input query and produce a structured QueryAnalysis object.
        """
        raw_query = query.strip()
        normalized = self.normalizer.normalize(raw_query)
        intent = self.classifier.classify(normalized)
        college_intent = self.classifier.classify_college_intent(normalized)
        target_cats = self.classifier.extract_target_categories(normalized, college_intent)
        target_depts = self.classifier.extract_target_departments(normalized)

        is_complex, factors, subqueries, comparison_obj = self.decomposer.decompose(
            original_query=raw_query,
            normalized_query=normalized,
            intent=intent,
        )

        return QueryAnalysis(
            original_query=raw_query,
            normalized_query=normalized,
            primary_intent=intent,
            college_intent=college_intent,
            target_categories=target_cats,
            target_departments=target_depts,
            is_complex=is_complex,
            complexity_factors=factors,
            subqueries=subqueries,
            comparison_objective=comparison_obj,
        )
