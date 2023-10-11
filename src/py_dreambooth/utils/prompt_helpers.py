import random


def make_prompt(subject_name: str, class_name: str) -> str:
    """
    Make a prompt sample
    Args:
        subject_name: The subject name used for training
        class_name: The class name used for training
    Returns:
        The prompt string
    """
    prompt_list = [
        "A hyper-realistic and stunning depiction of {subject_name} {class_name}, capturing the {class_name}'s charisma and charm, trending on Behance, intricate textures, vivid color palette, reminiscent of Alex Ross and Norman Rockwell",
        "A portrait of {subject_name} {class_name}, pen and ink, intricate line drawings, by Craig Mullins, Ruan Jia, Kentaro Miura, Greg Rutkowski, Loundraw",
        "A drawing of {subject_name} {class_name}, in the style of mark lague, hyper-realistic portraits, Sam Spratt, Brent Heighton, captivating gaze, cyclorama, crisp and clean --ar 69:128 --s 750 --v 5. 2",
        "A digital painting of {subject_name} {class_name}, a digital painting, magenta and gray, high contrast illustration, Ryan Hewett, Otto Schmidt",
        "A ultradetailed painting of a {subject_name} {class_name} by Conrad Roset, Greg Rutkowski, trending on Artstation",
        "Closeup of {subject_name} {class_name} posing in front of a solid dark wall, side profile, uhd, Kodak ektochrome, lifelike and stunning, cinematic light, volumetric light, Rembrandt lighting",
        "A professional photograph of of {subject_name} {class_name}, portrait photography, with detailed skin textures, shallow depth of field, Otus 85mm f/1. 4 ZF. 2 Lens, ISO 200, f/4, 1/250s, 8k --ar 2:3 --no blur, distortion, mutation, 3d, 2d, illustration",
        "A {subject_name} {class_name} with refined features and a sophisticated demeanor poses against a timeless backdrop. The 4K resolution highlights every aspect of the {class_name}'s elegance and charm, elegance-themed photography by Eleanor Eleganza, 2023, 4K camera, soft studio lighting",
        "Masterpiece, (beautiful and aesthetic:1. 5), surrealism, highly detailed, a portrait painting of {subject_name} {class_name}, hard brush, minimalist low poly collage illustration, splatter oil paintings effect, city portraits, heavy inking",
        "A drawing of a {subject_name} {class_name}, black and white, hints of oil painting style, hints of watercolor style, brush strokes, negative white space, crisp, sharp, textured collage, layered fibers, post-impressionist, hyper-realism",
    ]
    random.shuffle(prompt_list)
    for prompt in list(
        map(
            lambda x: x.format(subject_name=subject_name, class_name=class_name),
            prompt_list,
        )
    ):
        yield prompt


def make_negative_prompt() -> str:
    """
    Make a negative prompt sample
    Returns:
        The negative prompt string
    """
    negative_prompt = """
    (deformed iris, deformed pupils), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, 
    duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, 
    deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, 
    gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, 
    too many fingers, long neck
    """
    return ", ".join(map(lambda x: x.strip(), negative_prompt.split(",")))
