# ImproveLLMStorytellingWithFineTuning


## Knowledge Distillation for Storytelling Models

### Overview
Knowledge distillation is a technique used to transfer knowledge from a large, complex model to a smaller, more efficient one. In this project, we aim to distill storytelling knowledge from the larger language model, Llama 2-70B, to the smaller model, Phi 1-1B. This process will help us create a robust, efficient storytelling model that can generate high-quality narratives without the computational overhead of its larger counterpart.

### Strategy
The distillation process is broken down into two primary steps, focusing on generating and refining story content:

#### Step 1: Generating Story Topics
First, we provide the larger model, Llama 2-70B, with various story themes selected from the top 200 common literature themes listed by ProWritingAid, carefully filtering out any themes that involve explicit or restricted content. The model uses these themes to create concise story topics or summaries.

##### Prompt Engineering for Story Topics
We employ a specific prompting technique known as "1-shot learning" to guide the model in generating story summaries. This technique involves providing the model with a single example of the desired task, setting the context for its responses. Here’s how the prompt is structured:

```python
def create_topics_prompt(theme: str) -> str:
    """
    Generates a structured prompt for creating story topics based on a given theme.

    Args:
        theme (str): The specific theme for story creation.

    Returns:
        str: A fully formatted prompt for generating multiple story topics.
    """

    prompt = f'''
    [INST]:<<SYS>>You are a chatbot that provides story topics for users.
    
    For example:
    <User>: Please give me a summary of a story.
    
    Response:
    
    1. An ugly dragon seeking the crystal of the forest.
    2. A fairy with a desire to feast on carnivorous plants.
    3. A space mermaid carrying a fan.
    4. A mummy harboring dreams of becoming a Hollywood star.
    5. A samurai who fell in love with a fish.
    
    Now it's your turn. Please respond in English in a creative and innovative manner. The more unique and different, the better.
    Your answer must be in topics like above. Do not give titles for your stories!<<SYS>>
    
    <<User>>: Invent a completely new story summary about "{theme}". Provide 7 possibilities. [/INST]
    '''

    return prompt

```


The model will use these themes to create concise story topics or summaries that encapsulate potential narratives. For example, given the theme "Abuse of power," the model might generate a topic like:

```python
"The Last Days of the Solar Empire: A tale of a dying empire ruled by a mad emperor who uses his power to build a pyramid of skulls."
```

These topics serve as a distilled essence of potential stories, providing a thematic and narrative foundation without detailing the full story.

#### Step 2: Expanding into Full Stories

Next, we take the generated topics from Step 1 and use them as prompts for the smaller model, Phi 1-1B, to develop into full-length stories. This approach allows the smaller model to focus on narrative expansion, characterization, and detail without the initial overhead of theme conceptualization. For instance, the topic about the Solar Empire would be expanded into a detailed story, illustrating the decline of the empire and the emperor's tyranny as follows:



