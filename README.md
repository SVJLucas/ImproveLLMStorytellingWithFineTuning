
# GDSC AI Workshop: Enhancing Storytelling in Language Models through Fine-Tuning and Knowledge Distillation



## About the Workshop

<p align="center">
  <img width="500" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/SVJLucas/SVJLucas/assets/60625769/1501cb07-3add-4907-9778-10b90c223e69">
</p> 

This workshop is organized and led by [Lucas José](https://www.linkedin.com/in/lucasjosevelosodesouza/), a leader in the Google Developer Student Clubs (GDSC). It is tailored for students eager to explore the field of Artificial Intelligence, with a specific focus on Natural Language Processing (NLP). The primary aim is to develop practical skills in fine-tuning smaller Language Models (LLMs) to enhance their storytelling capabilities.


## Objective
The primary goal of this workshop is to empower participants with the ability to implement knowledge distillation and fine-tuning techniques to create a compact yet powerful storytelling model. By distilling the capabilities of a larger model (Meta Llama 2-70B) into a smaller model (Microsoft Phi 1-1B), participants learn how to maintain high-quality narrative generation while reducing computational demands, making advanced NLP technologies more accessible and efficient.

<p align="center">
  <img width="400" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/100636f1-8a76-4bb3-987d-e8621971698d">
</p> 

### Workshop Program
The workshop covers a range of topics, including:

* **Introduction to Knowledge Distillation**: Participants are introduced to the concept of knowledge distillation, where knowledge from a larger model is transferred to a smaller model without significant loss in performance.
* **Practical Fine-Tuning Session**: Attendees participate in hands-on sessions where they apply fine-tuning techniques on Phi 1-1B using the QLORA (Quantized Linear Operation with Regularization and Attention) method, which is crucial for managing resource constraints while maintaining model effectiveness.
* **Dataset Creation and Management**: The workshop also guides participants through the creation of a storytelling dataset. This involves generating story topics from a curated list of literature themes and expanding these topics into full narratives.
* **Fine-Tuning in High-Performance Environments**: A detailed session on how to set up and run fine-tuning processes in _Google Colab_ on an A100 GPU, utilizing specific configurations to optimize memory and compute resources.



## Knowledge Distillation for Storytelling Models

### Overview
Knowledge distillation is a technique used to transfer knowledge from a large, complex model to a smaller, more efficient one. In this project, we aim to distill storytelling knowledge from the larger language model, Llama 2-70B, to the smaller model, Phi 1-1B. This process will help us create a robust, efficient storytelling model that can generate high-quality narratives without the computational overhead of its larger counterpart.

### Strategy
The distillation process is broken down into two primary steps, focusing on generating and refining story content:

#### Step 1: Generating Story Topics
First, we provide the larger model, Llama 2-70B, with various story themes selected from the top 200 common literature themes listed by ProWritingAid, carefully filtering out any themes that involve explicit or restricted content. The model uses these themes to create concise story topics or summaries that represent the main story ideas.

<p align="center">
  <img width="700" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/1de63871-9f89-4e19-8ba0-051c4d7573ad">
</p> 

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

After generating story topics in Step 1, we utilize these prompts to guide the smaller model, Phi 1-1B, in crafting detailed, narrative-driven stories. This structured approach allows Phi 1-1B to concentrate on narrative development, enriching the plot and characters without the initial burden of theme conceptualization.

<p align="center">
  <img width="700" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/362c43bb-391a-4621-a223-b2d23c6e42dc">
</p> 

##### Prompt Engineering for Stories

To aid the smaller model in creating compelling narratives, we use a prompt engineering technique that specifies the desired structure and brevity of the stories. Here's the function used to generate these prompts:

```python
def create_story_prompt(topic: str) -> str:
    """
    Constructs a prompt for generating a short story focused on a specific topic, ensuring the story is concise yet complete.

    Args:
        topic (str): The central theme of the story to be created.

    Returns:
        str: A formatted prompt instructing the model to produce a short story with two compact paragraphs, including a resolution and character development.
    """
    # Define the prompt with specific instructions for story creation
    prompt = f'''
    [INST]:<<SYS>>You are a chatbot tasked with creating stories for users.
    
    Your stories must be short, they CANNOT be long, which means your stories must have only TWO paragraphs of small size.
    Ensure each story has a clear conclusion and well-defined characters!<<SYS>>
    
    <<User>>: Invent a completely new story about "{topic}".[/INST]
    '''
    return prompt
```


For instance, the topic about the Solar Empire would be expanded into a detailed story, illustrating the decline of the empire and the emperor's tyranny as follows:

```python
"In the final days of the Solar Empire, a mad emperor ruled with an iron fist. His power was unmatched, and his people lived in fear of his wrath. The emperor was obsessed with the idea of building a pyramid of skulls, a monument to his own greatness. He believed that the pyramid would grant him immortality, and he spared no expense in its construction.\n\nThe empire was in decline, and the people suffered under the emperor's rule. They were forced to work long hours, and their meager wages were barely enough to sustain them. The emperor cared little for their suffering, and he taxed them heavily to fund his grand project. As the pyramid rose higher and higher, the people grew more and more desperate. They knew that the emperor's madness would be their downfall, and they longed for the day when he would be overthrown. But the emperor's grip on power was too strong, and it seemed that he would never be defeated. The pyramid of skulls stood as a symbol of the empire's decay, a monument to the emperor's madness and the people's suffering."
```

The final dataset has the amount of **1476** stories, and with them we'll finetunning out small LLM.



## Running the Model Fine-Tuning in Google Colab on V100 GPU

### Utilizing QLORA for Fine-Tuning

<p align="center">
  <img width="500" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/833c902e-c644-4441-8126-c3ade8c5a8e1">
</p> 

QLORA (Quantized Linear Operation with Regularization and Attention) is an advanced technique used in our fine-tuning process. This method involves modifying parts of the transformer model to use quantized computations, which significantly reduces the memory footprint without compromising the model's ability to learn complex patterns. By using QLORA, we can train larger models or use larger batch sizes within the same memory limits, enhancing the effectiveness of our training procedure on resource-restricted platforms like Google Colab.


<p align="center">
  <img width="500" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/5bd7fdab-0cec-4606-996b-99da8dad9c69">
</p> 



### Fine-tuning Parameters Details


<p align="center">
  <img width="350" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/92a68066-0e31-442d-aa9c-c6bab936a31c">
</p> 


To successfully run the fine-tuning process on a V100 GPU in Google Colab, specific training configurations are essential to maximize efficiency and fit the model within the available memory constraints. Below is an explanation of each configuration parameter and its significance:


* **Batch Size and Gradient Accumulation:** Given the memory constraints of GPUs, we use a smaller batch size per device. To compensate for the smaller batch size and still leverage the deep learning benefits of larger batches, we implement gradient accumulation. This approach means that the model updates its weights only after several forward passes, effectively simulating a larger batch size.

* **Learning Rate and Scheduler:** A lower learning rate combined with a cosine learning rate scheduler ensures smooth and effective model training over epochs. The scheduler adjusts the learning rate following a cosine curve, reducing the risk of overshooting minima in the loss landscape.

* **Warmup Steps:** Implementing a few warmup steps at the beginning of the training helps in gradually ramping up the learning rate from zero. This technique is beneficial for stabilizing the training early on, preventing the model from diverging due to high initial learning rates.

Incorporating these configurations and techniques ensures that the fine-tuning process is not only memory-efficient but also robust and effective, making it possible to train state-of-the-art models even in constrained environments like Google Colab.

