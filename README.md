# ImproveLLMStorytellingWithFineTuning


## Knowledge Destillation

We are going to Destil the Storytelling Knowledge from the bigger LLM (Llama 2-70B) to the smallest one (Phi 1-1B) creating a Dataset in two steps:

1. We'll give the biggest LLM stories themes and will ask them to create a small summary/topic of  a possible storie inside the theme. For exemple, for the theme 'Abuse of power' an possible Topic is:

```
The Last Days of the Solar Empire: A tale of a dying empire ruled by a mad emperor who uses his power to build a pyramid of skulls
```

2. After we create the possible topics/summaries of the stories, we can ask for the model generate a longer story of that topic. In the case of the topic above, this would be:

```
In the final days of the Solar Empire, a mad emperor ruled with an iron fist. His power was unmatched, and his people lived in fear of his wrath.

The emperor was obsessed with the idea of building a pyramid of skulls, a monument to his own greatness. He believed that the pyramid would grant him immortality, and he spared no expense in its construction.

The empire was in decline, and the people suffered under the emperor's rule. They were forced to work long hours, and their meager wages were barely enough to sustain them.

The emperor cared little for their suffering, and he taxed them heavily to fund his grand project. As the pyramid rose higher and higher, the people grew more and more desperate.

They knew that the emperor's madness would be their downfall, and they longed for the day when he would be overthrown.

But the emperor's grip on power was too strong, and it seemed that he would never be defeated. The pyramid of skulls stood as a symbol of the empire's decay, a monument to the emperor's madness and the people's suffering.
```

