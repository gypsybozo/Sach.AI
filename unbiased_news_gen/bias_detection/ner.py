import spacy
import logging
from unbiased_news_gen import config
from unbiased_news_gen.data_ingestion import loader
from thefuzz import fuzz # Use thefuzz library
from collections import OrderedDict

# Load spaCy model
try:
    nlp = spacy.load(config.SPACY_MODEL)
except OSError:
    logging.error(f"SpaCy model '{config.SPACY_MODEL}' not found.")
    nlp = None

# Load entity definitions AND the lookup dictionary
entity_df, entity_lookup = loader.load_entity_definitions()
# Fallback if lookup fails
if entity_lookup is None:
    logging.warning("Failed to create entity lookup dictionary. Category information will be missing.")
    entity_lookup = {}
CUSTOM_ENTITIES_LOWERCASE = set(entity_lookup.keys()) # Get keys from the lookup
logging.info(f"Loaded {len(CUSTOM_ENTITIES_LOWERCASE)} unique lowercase custom entities for NER.")

def extract_entities_enhanced(text):
    """
    Extracts named entities using spaCy and enhances with fuzzy matching
    against a custom list, attempting to resolve overlaps.
    """
    if nlp is None or not isinstance(text, str):
        return []

    doc = nlp(text)
    found_entities = OrderedDict() # Use OrderedDict to preserve order and handle overlaps

    # 1. SpaCy NER Pass
    for ent in doc.ents:
        key = (ent.start_char, ent.end_char)
        entity_lower = ent.text.lower()
        # --- Add category lookup ---
        category = 'Unknown' # Default
        is_custom = False
        if entity_lower in entity_lookup:
             is_custom = True
             category = entity_lookup[entity_lower].get('category', 'Unknown')
        # --- End category lookup ---
        found_entities[key] = {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "source": "spacy",
            "is_custom": is_custom,
            "category": category # Add category here
        }

    # 2. Fuzzy Matching Pass for Custom Entities
    text_lower = text.lower()
    for custom_ent_lower in CUSTOM_ENTITIES_LOWERCASE:
        # Skip if already found by spaCy via exact match
        already_found_exact = any(
            data['text'].lower() == custom_ent_lower and data['source'] == 'spacy'
            for data in found_entities.values()
        )
        if already_found_exact:
            continue

        search_start = 0
        while search_start < len(text_lower): # Ensure we don't go past the end
            # Find first letter of custom entity in the remaining text
            start_index = text_lower.find(custom_ent_lower[0], search_start)
            if start_index == -1:
                break # First letter not found in the rest of the text, move to next custom entity

            # Define a window around the potential start
            window_end = min(len(text), start_index + int(len(custom_ent_lower) * 1.5) + 10) # Extend window slightly more
            text_window = text[start_index:window_end]
            if not text_window: # Skip empty windows
                 search_start = start_index + 1
                 continue

            # Use token_set_ratio for fuzzy match
            match_score = fuzz.token_set_ratio(custom_ent_lower, text_window.lower())

            # --- Check match score BEFORE defining potential_key and checking overlaps ---
            if match_score >= config.FUZZY_MATCH_THRESHOLD:
                # --- This whole block only runs if a good fuzzy match is found ---

                # Simple Approximation for boundaries (Refine if needed)
                best_match_text = text_window # Placeholder, Ideally find exact span
                end_char_approx = start_index + len(best_match_text)

                potential_key = (start_index, end_char_approx)

                # Overlap Check
                is_overlapping = False
                entity_to_update = None
                for (e_start, e_end), e_data in found_entities.items():
                    # Check for overlap: !(end1 <= start2 or start1 >= end2)
                    if not (potential_key[1] <= e_start or potential_key[0] >= e_end):
                        is_overlapping = True
                        # If overlaps with spaCy, maybe mark spaCy one as custom & add category
                        if e_data['source'] == 'spacy' and not e_data['is_custom']:
                            if fuzz.token_set_ratio(custom_ent_lower, e_data['text'].lower()) > config.FUZZY_MATCH_THRESHOLD - 5:
                                entity_to_update = (e_start, e_end) # Mark which spacy entity to update
                        break # Stop checking overlaps for this potential fuzzy match

                # Update existing spaCy entity if needed (outside the inner loop)
                if entity_to_update:
                    found_entities[entity_to_update]['is_custom'] = True
                    found_entities[entity_to_update]['category'] = entity_lookup[custom_ent_lower].get('category', 'Unknown')
                    logging.debug(f"Marked overlapping spaCy entity '{found_entities[entity_to_update]['text']}' as custom (Category: {found_entities[entity_to_update]['category']}) based on fuzzy match with '{custom_ent_lower}'")

                # Add fuzzy match only if it doesn't significantly overlap *AND* isn't already added
                # (The entity_to_update logic handles overlap with spaCy; now check if adding non-overlapping)
                if not is_overlapping:
                    if potential_key not in found_entities:
                        fuzzy_category = entity_lookup[custom_ent_lower].get('category', 'Unknown')
                        found_entities[potential_key] = {
                            "text": best_match_text,
                            "label": "CUSTOM_ENTITY", # Use a distinct label
                            "start_char": start_index,
                            "end_char": end_char_approx,
                            "source": "fuzzy",
                            "is_custom": True,
                            "category": fuzzy_category
                        }
                        logging.debug(f"Added fuzzy match: '{best_match_text}' (Category: {fuzzy_category}) for custom entity '{custom_ent_lower}'")

                # If we found a good match starting at start_index,
                # advance search_start past this match area for the *next* iteration of the while loop
                search_start = start_index + 1 # Could be improved to jump past end_char_approx
                # Decide if you want to find *all* possible fuzzy matches or just the first good one starting from an index
                # break # Uncomment this line if you only want the first good fuzzy match per custom entity starting point

            else:
                # If match score is too low, just advance search_start
                search_start = start_index + 1
        # --- End of while loop ---
    # --- End of for custom_ent_lower loop ---

    final_entities = sorted(list(found_entities.values()), key=lambda x: x['start_char'])
    # TODO: Add post-processing step here if needed to merge/filter overlapping entities based on score/source/length preference
    return final_entities