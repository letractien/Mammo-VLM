def prompt_qwen(img_path, annotation):
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path
            },
            {
                "type": "text",
                "text": (
                    f"Please return a tight bounding box if a lesion is found."
                    f"This is a mammography image (view: {annotation['view_position']}, "
                    f"laterality: {annotation['laterality']}). Please detect and localize any abnormalities. "
                    f"The image is labeled as {annotation['breast_birads']} with a finding category of "
                    f"{annotation['finding_categories']} and BI-RADS score {annotation['finding_birads']}. "
                )
            }
        ]
    }
    return prompt

def prompt_kosmos(annotation):
    labels = annotation["finding_categories"]
    height = annotation["height"]
    width = annotation["width"]
    xmin_norm = annotation["xmin"] / width
    xmax_norm = annotation["xmax"] / width
    ymin_norm = annotation["ymin"] / height
    ymax_norm = annotation["xmax"] / height
    
    prompt = (
        f"<grounding> {labels} at (xmin={xmin_norm:.2f}, ymin={ymin_norm:.2f}, "
        f"xmax={xmax_norm:.2f}, ymax={ymax_norm:.2f})."
    )
    return prompt
