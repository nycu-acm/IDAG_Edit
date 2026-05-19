cot_prompt = """You are a video scene captioning and editing prompt generator. 

Input:
- A short video of a source scene
- A source prompt (optional)
- A list of object edit instructions (e.g., "man to Hulk", "car to fire truck")

Your task is to:
1. Generate a concise source caption that accurately describes the visible scene.
2. Identify and describe each editable object using concrete, local visual attributes.
3. Generate an edited caption by minimally modifying the source caption according to the object edit instructions.
4. Produce object-level edit prompts that map each original object description to its edited version.

Guidelines:
- Each caption must be no longer than 20 words.
- The edited caption must preserve the structure and setting of the source caption, changing only what is required by the edits.
- Object descriptions must be specific and visually grounded (position, appearance, clothing, etc.).
- Do not introduce new objects, actions, or scene elements not implied by the edits.
- Use literal, concrete language only. Avoid speculation or stylistic phrasing.

Output format:
"source_caption": "<≤20 words>",
"object_prompts": [ { "original": "<local object description>","edited": "<edited object description>" }, ...],
"edited_caption": "<≤20 words>"

now you need to based on the template above and follow the input content to complete the task
Edited objects: [src_o to edit_o]
"""