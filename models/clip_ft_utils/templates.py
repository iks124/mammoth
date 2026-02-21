from utils.prompt_templates import templates as dataset_templates


imagenet_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]


dataset_to_template = {
    "seq-eurosat-rgb": dataset_templates.get("eurosat", imagenet_template),
    "seq-resisc45": dataset_templates.get("resisc45", imagenet_template),
    "seq-cropdisease": dataset_templates.get("cropdisease", imagenet_template),
    "seq-imagenet-r": imagenet_template,
}


def get_templates(dataset_name):
    if dataset_name.endswith("Val"):
        return get_templates(dataset_name.replace("Val", ""))
    if dataset_name in dataset_to_template:
        return dataset_to_template[dataset_name]
    return imagenet_template
