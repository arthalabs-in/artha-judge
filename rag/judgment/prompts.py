JUDGMENT_EXTRACTION_PROMPT = """
Extract court judgment facts as strict JSON. Every value must include source
evidence and confidence. If the text does not support a field, return null.
"""

ACTION_PLAN_PROMPT = """
Convert verified judgment directions into government action items as strict
JSON. Do not invent departments, deadlines, or legal obligations.
"""
