import random


def make_prompt(subject_name: str, class_name: str) -> str:
    prompt_list = [
        "A hyper-realistic and stunning depiction of {subject_name} {class_name}, capturing the person's charisma and charm, trending on Behance, intricate textures, vivid color palette, reminiscent of Alex Ross and Norman Rockwell",
        "A portrait of {subject_name} {class_name}, pen and ink, intricate line drawings, by Craig Mullins, Ruan Jia, Kentaro Miura, Greg Rutkowski, Loundraw",
        "A drawing of {subject_name} {class_name}, in the style of mark lague, hyper-realistic portraits, Sam Spratt, Brent Heighton, captivating gaze, cyclorama, crisp and clean --ar 69:128 --s 750 --v 5. 2",
        "A painting of {subject_name} {class_name}, in the style of Yuumei, dark orange and gray, depictions of inclement weather, detailed character design, Frank Miller, angular, Eddie Jones --ar 69:128 --s 750 --v 5. 2",
        "A digital painting of {subject_name} {class_name}, a digital painting, magenta and gray, high contrast illustration, Ryan Hewett, Otto Schmidt",
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
    negative_prompt = """
    (deformed iris, deformed pupils), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, 
    duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, 
    deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, 
    gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, 
    too many fingers, long neck
    """
    return ", ".join(map(lambda x: x.strip(), negative_prompt.split(",")))
