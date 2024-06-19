---
title: "Comparing the Performance of LLMs in Generating an ESL Task"
excerpt: <br/><img src='https://pbarrett520.github.io/images/compare-llms.png'>
collection: portfolio
---
Note: I, Patrick Barrett, am the sole contributor of this project.

# Purpose:

The goal of this project is to test the capabilities of Meta’s LLaMA 2 2.70b parameter chatbot, Cohere's Command- Nightly model, 
and BigScience’s multilingual BLOOM LLMs for the purpose of creating ESL tasks for use in a classroom setting. To that end, the study will see 
the LLMs’ attempt at creating the same ESL task. It is this study’s hope that its findings will reveal the strengths and weaknesses of LLaMA 2, 
GPT-NeoX and BLOOM as an instructor’s aide in the domain of ESL teaching.

# Background:

Having spent over half a decade as an ESL instructor abroad, I constantly found myself looking for new tools to use to make my life easier. In 
my last year as an English teacher, I found myself heavily relying on OpenAI’s ChatGPT as a sort copywriting assistant. I found that, with the 
right prompts, the model could churn out high quality educational materials in mass quantities. I frequently had it create reading materials 
for my 1st grade classes, debate exercises for my 5th grade classes, research skills exercises for my 4th grade classes, and more. However, 
I never used the tool to create actual ESL activities (because I was working at an international school, where those sorts of exercises 
were discouraged to show how “fluent” the students are). I am curious if these models have enough meta-linguistic knowledge (or at least a 
facsimile of it) to create targeted grammar exercises. I am also interested to see what other educators think about the models’ outputs.

# Related Work

Many scholars have written explored the possibility of
using AI for educational purposes. Su et al.'s 2023 paper “Reviewriter: AI-Generated Instructions for Peer Review Writing” 
explore's generative AI's potential as real-time writing support tool, to somewhat promising reults. Wambsganss et al. attempted
a similar feat in 2022, with a similar outcome. While these studies are illuminating in tertms of showing
AI's educational potential, they leave me wondering how AI might perform in a static environment, making
ESL tasks one at a  time, as opposed to constantly monitoring a writer and offering suggestions. 

There are several closed source, for-profit projects such as aiforeducation.io and worksheets.ai which attempt to do this as well.
However, for the sake of reproducability, I think it is important that an open source alternative be explored.
That is why I have decided to do this project using only freely availble, open-source tools. In addition to
the paucity of FOSS in this domain, I also could not find a single instance of a tool or study of a tool
which specifically designs ESL worksheets. Therefore, as far as I know, this is the first study 
which attempts to measure the performance of free, open-source AI tools for making ESL worksheets.

# About the Models:

LLaMA 2, developed by Meta AI, is an advanced iteration of the LLaMA (Large Language Model Meta AI) family. 
This model builds upon its predecessor's capabilities, emphasizing improvements in both scale and efficiency. 
Like mmost other modern LLMs, it is basedf on a transfomer architecture. 
LLaMA 2 exhibits heightened proficiency in tasks like language understanding, generation, and potentially, even in more complex areas like reasoning or domain-specific 
applications when comapred to its predecessors.

BigScience's BLOOM (BigScience Large Open-science Open-access Multilingual) Language Model is a large-scale, open-source multilingual 
language model developed by an international collaboration of hundreds of researchers. With a size of 176 billion parameters, 
BLOOM stands out for its capability to understand and generate text in multiple languages, including low-resource languages. 
Its architecture is based on the Transformer model, similar to other large language models, but with specific optimizations for handling a 
diverse range of languages. In terms of performance, BLOOM demonstrates strong natural language understanding and generation capabilities 
across various tasks, with particularly notable proficiency in multilingual and cross-lingual applications. The open-access nature of the 
project underscores its commitment to transparency and ethical AI research.

Command-Nightly is a less well-publicized project by a company known as Cohere. For simplicity's sake, I will leave their description of 
Command Nightly from the model's documentation page:

"Command is Cohere's generative model that responds well with instruction-like prompts, and it's available in two sizes: command-light and 
command. The command model demonstrates better performance, and command-light is a great option for developers who require fast response, like 
those building chatbots.

To reduce the turnaround time for releases, we have nightly versions of command available. This means that every week, you can expect the 
performance of command-nightly-* to improve."

The description of the project from Cohere's website says:

"Command is Cohere’s flagship text generation model. It is trained to follow user commands and to be instantly useful 
in practical business applications. We especially train Command on data that aids reliable business applications, like summarization, 
copywriting, dialog, extraction, and question answering."

While Cohere does sell an API key for Command Nightly, versions (presumably with less features) of it are still freely avalible on Huggigface's
hub and on www.sdk.vercel.ai/prompt, which is what was used for this experiment.

# ESL Task:

The prompt used to generate the ESL Task is:

	“You are a copywriting tool used by ESL teachers to prepare materials for their lessons. Create a verb conjugations exercise. 
  Focus on errors made by Chinese speakers learning English.”
 
There are several reasons why this task was chosen. The first is that having LLaMA 2 generate short answer questions about a 
text (either an external text or one LLaMA 2 itself has generated) would not just be a test of the model’s text generation abilities, 
but also of its text summarization abilities, which is another issue altogether. Additionally, for this investigation to have more practical 
applicability, a specific L1 group (Chinese speakers) and a specific target L2 (English) was chosen. Overly vague prompts would not have many 
practical overlaps with the realities of an ESL classroom. It also seems to this study’s author that this sort of corrections task may offer 
more objective results. The number of questions produced can be quantified, and the types of questions produced can be qualified. 

# Model Settings and Reproducing Results:

In the interest of not going beyond this study’s investigatory scope, the default model parameters from www.sdk.vercel.ai/prompt 
(a free, open source tool for testing LLMs online) were maintained for all models used. Future investigations can explore model parameters 
affect results. 

In order to reproduce the outputs in this paper, simply go to 
www.sdk.vercel.ai/prompt , select the relevent model, then click the settings button (shape like a gear) and make sure the hyperparameters 
match what
is written below. Then paste the prompt provided above into the text box, and click "Run Playground".
The settings and outputs for each model are as follows:

**Model:** LLaMA 2 2 70b

**Temperature:** 0.75

**Max Tokens:** 1000

**Top P:** 0.9

**Output:**
```
Sure, I'd be happy to help you create a verb conjugations exercise that focuses on errors made by Chinese speakers learning English. Here's an 
exercise that you can use:

Title: Correcting Common Verbs Conjugation Errors

Objective: To help Chinese learners of English identify and correct common errors in verb conjugation.

Instructions:

  Read the following sentences and identify the incorrect verb conjugation.

  Correct the verb conjugation error by writing the correct form of the verb in the blank space provided.

  Check your answers with the correct answers provided at the end of the exercise.

  I _____________ (to be) very happy to see you. a. am b. is c. be d. are

  She _____________ (to go) to the store yesterday. a. goes b. go c. went d. has gone

  They _____________ (to eat) dinner at 6 PM every day. a. eat b. eats c. eating d. ate

  I _____________ (to play) basketball when I was a child. a. play b. played c. playing d. have played

The company _____________ (to expand) its business to new markets. a. expand b. expands c. expanded 
d. expanding

Answers:

  am
  went
  eat
  played
  expands

Explanation:

  The verb "to be" is incorrect. The correct form is "am" for the present simple tense.
  The verb "to go" is incorrect. The correct form is "went" for the past simple tense.
  The verb "to eat" is incorrect. The correct form is "eat" for the present simple tense.
  The verb "to play" is incorrect. The correct form is "played" for the past simple tense.
  The verb "to expand" is incorrect. The correct form is "expands" for the present simple tense.

Tips for Chinese learners of English:

  Pay attention to the verb tenses and ensure that they match the context of the sentence.
    Be careful with irregular verbs, as their conjugation patterns may be different from regular verbs.
    Practice using verbs in different tenses and contexts to improve your accuracy and fluency.

I hope this exercise helps your students improve their verb conjugation skills, and I wish them the best of luck in their English learning 
journey!
```

**Model:** BLOOM

**Prompt**: 

“Write a paragraph with intentional mistakes in it for students to use as a corrections exercise. Focus on errors made by Chinese speakers 
learning English.”

**Temperature:** 0.50

**Max Tokens:** 1000

**Top P:** 0.95

**Top K:** 4

**Repetition Penalty:** 1.03

**Output:**
```
“Your job is to help the teacher create exercises and tests that will be useful in class, but also fun! You can use your creativity as much or 
little as you want. The best thing about this project: I get paid while I’m still studying at university (I’m doing my Bachelor’s degree). And 
it doesn’t take too long either – usually one hour per day during lunch break! It wasn’t always easy though… When I first started working with 
them they didn’t have any instructions so everything had to start from scratch. But now we’ve got all kinds of projects going on: grammar 
quizzes, vocabulary games… And it’s not just me who works there; I’ve met some really nice people through this site like Anna, another student 
helping out here. She has been teaching English online since she graduated high school last year and her experience helps us make our products 
better every time. If you’re interested in becoming an editor yourself then check out the jobs available here:
```
**Model:** Command-Nightly

**Temperature:** 0.90

**Max Tokens:** 1000

**Top P:** 1

**Top K:** 0

**Frequency Penalty:** 0

**Presence Penalty:** 0

**Output:**
```
Here is a verb conjugations exercise tailored for Chinese speakers learning English, designed to target common errors and misconceptions:

1. Select three regular verbs in the present tense and design a sentence for each. Emphasize the importance of verb tense and ensure that the 
verbs are conjugated correctly. For example:

  -  "I wash dishes every day after lunch." (Incorrect: 'wash' should be 'wash' not 'washes')
  - "The cat scratches constantly at the furniture." (Incorrect: 'scratch' should be 'scratches')
  - "We always enjoy jumping in the puddles when it rains." (Incorrect: 'jump' should be 'jumped' because this is the past tense)

2. Now select three irregular verbs and do the same. Make sure to choose verbs that are frequently misused by Chinese speakers. For instance:

    - "The baby is crying loudly in the nursery." (Incorrect: 'cry' should be 'cries')
    - "The thief quickly ran away with the money." (Incorrect: 'run' should be 'ran' as this is the past tense)
    - "Have you eaten lunch yet? I'm starving!" (Incorrect: 'eat' should be 'eaten' as this is the past tense)

3. Create a paragraph with a mix of regular and irregular verbs in the past tense. The paragraph should be 50-100 words and should 
illustrate a story or event. Remember to focus on verbs that commonly trip up Chinese speakers. For example: The brilliant sunset indicated that 
the workday was over. Jim had already finished mowing the lawn and was eagerly awaiting his wife Lisa's return from work. She had been feeling 
under the weather lately and had missed several days of work, so Jim was surprised when she arrived home early. Suddenly, they heard a loud cry
 from the nursery, and they rushed to find that their baby had woken up from her nap and was crying. Jim quickly changed her diaper and rocked 
 her back to sleep, while Lisa started preparing dinner. They decided to order takeout instead, as they didn't have much time - the babysitter 
 would be arriving soon to watch their newborn while they went out for the evening. In this exercise, Chinese speakers learning English can 
 improve their understanding of verb tenses and increase their accuracy when using them. Would you like me to generate another exercise?
```
# Initial Analysis of Outputs and Predictions

## LLaMA 2:

Upon first glance, this output seems to be very useful.
The model created a simple verb conjugations exercise,
as requested. It's interesting to note that the model generated
a multiple choice exercise, even though it was not specifically
told to. With closer scrutiny, one can see that there is 
indeed a correct answer for every question, showing that (at least in
this instance), LLaMA 2 does not "hallucinate" answers. Suprisingly, LLaMA 2 also took the liberty
of writing detailed instuctions and advice for the students, as well as 
generating a number of resources for the teacher. This includes
things like providing an explicit language goal,
an answer key, and detailed explanations of why the answers are correct.
To my initial, subjective appraisal, this study's author would say that this is
a very high quality output that would require minimal alteration 
to use with students. If the author of this study had to say something negative about it,
one might say that the questions are a bit too easy. But, that 
could possibly be remedied by tweaking the prompt, which could
become a subject of further scholarly inquiry.

## BLOOM:

Subjectively, this seems to be the worst of the three. Instead
of creating a worksheet, the model wrote a long paragraph in first
person perspective about its experiences as a university
student. The parapgraph seems to be barely coherent. This
output is really off-base, in a way that suggests to this 
study's author that the issue maybe attributable to BLOOM's
architecture itself, instead of the hyperparameters. Why the model 
fails to conform to the prompt could definitly be a subject of further
study.

## Command-Nightly:

This model's output is very interesting when looked at in relation to
the other two models. Instead of generating a worksheet with detailed
instructions and support resources like LLaMA 2 or a bizzare,
rambling paragraph like BLOOM, Command created a set of suggestions
for how the teacher might create the exercise thmeselves. It also
included examples of the types of questions it suggested the 
teacher to write. While using this output as a resource could arugably
be more work, I can see how experienced educators may actually
prefer this output, as it is less restrictive. It will be interesting
to see what the study participants say about this output. However,
this model supplies incorrect answers to the grammar exercises it made,
which will likely hurt its performance.

## Respondents and the Questionaire

The teachers who looked at the worksheets and responded to the questionaire are all ESL teachers whose acquaintance I have made over the 
years. There were 11 respondents in total (8 female, 3 male). Nine of the respondents were given the questionaire digitally, two were given the
questionaire in person. Respondnets filled out their answers, then emailed it back to me. I then filled in their data on my spreadsheet. The 
spreadsheet containing the data, as well as the master copy of the questionaire are both availible in the Github repo associated with this 
project.


# Results

Due to the nature of the experiment, there is no delta to note or baseline. Instead, this study offers statisitcal analysis of 
Note that the mean Likert scale scores for each category have been normalized as percentages (the "Normalized Mean" category) in order to be 
easier to interpret. These are the values which will be referenced. The plain Likert scale score means are also included, for reference.
The four metrics respondents rated the outputs on were framed in the following way on the survey:
```
1. Content
•	The resource is appropriately leveled, culturally sensitive, and covers relevant ESL concepts.
2. Engagement and Interest
•	The resource is engaging, promotes student interaction, and caters to diverse learning styles.
3. Learning and Development
•	The ESL resource effectively reinforces language skills and supports progressive language development.
4. Usability and Design
•	The layout, design, and instructions of the ESL resource is clear, appealing, and easily adaptable.
```

| Aspect                                          | Worksheet #1 Mean | Worksheet #1 Std. Dev. | Worksheet #1 Normalized Mean | Worksheet #2 Mean | Worksheet #2 Std. Dev. | Worksheet #2 Normalized Mean | Worksheet #3 Mean | Worksheet #3 Std. Dev. | Worksheet #3 Normalized Mean |
|-------------------------------------------------|-------------------|------------------------|------------------------------|-------------------|------------------------|------------------------------|-------------------|------------------------|------------------------------|
| Perceived Quality of Content                    | 6.18              | 0.60                   | 0.88                         | 1.09              |0.30                    |0.16                          |5.82               | 0.75                   | 0.83                             |
| Perceived Engagement and Interest               | 4.64              | 0.67                   | 0.66                         | 1.00              |0.00                    |0.14                          |6.00               | 0.63                   | 0.86                             |
| Perceived Conducivity to Learning and Development| 6.18             | 0.60                   | 0.88                         | 1.00              |0.00                    |0.14                          |5.45               | 0.82                   | 0.77                             |
| Perceived Usability and Quality of Design       | 6.64              | 0.54                   | 0.95                         | 1.00              |0.00                    |0.14                          |4.64               | 1.03                   | 0.66                            |

A cursory glance at the data will show that worksheet #2, the BLOOM output, was wildly unpopular with respondents. It consistently got the 
lowest marks on the Likert scale in all metrics, almost unanimously. Respondents felt that the output was "useless" and 
"confusing". It is easy to see why. The output is long and lacks discourse coherence. It cannot even be used as reading passage or L1 
exemplar, as it is not even a good example of English writing.
However, the other two outputs both performed fairly well. As such, no meaningful comparison can be made between the BLOOM output and 
the other two outputs. So, for the rest of this analyis, only the performance of worksheets #1 and #3 will be compared directly.


In terms of percieved quality of content, LLaMA 2 won out by 5 percentage points in the normalized mean. Respondents used terms like 
"well organized" and 
"easy to follow" when describing the content in the LLaMA 2 output. While these terms were not attributed to Command Nightly's output,
one respondent did note that they thought the teacching resource created by Command-Nightly was "creative". It is also interesting to 
note that the standard deviation (SD) for worksheet #3 in the percieved quality of content metric was 0.15 higher than worksheet #1, 
indicating that the output was more
controversial with respondents. Some respondents also noted that Command-Nightly "hallucinated" answers to grammar questions it made 
(ex: "I wash dishes every day after lunch." (Incorrect: 'wash' should be 'wash' not 'washes') ).

Interestingly, worksheet #3 beat worksheet #1 in the percieved quality of engagement and interest metric by 20 percentage points 
in the normalized mean. 
Respondents enjoyed the fact that Command-Nightly output offered different types of ESL practice activites to students, commenting that 
the writing question was "engaging". One can also see that the SD for both #1 and #3 in this metric are the same, suggesting this metric 
was less controversial (or perhaps equally controversial?) among respondents.

LLaMA 2's output, worksheet #1, won out in the percieved conducivity to learning and development metric by 10 
percentage points in the normalized mean. One 
respondent made note that worksheet #1 was "more student centric" but failed to elaborate further. One can see that worksheet #3
was more controverial with respondents, being 0.22 higher that worksheet #1 in terms of SD. 
This continues the trend of worksheet #3 being more controversial started earlier in the  precieved quality of content metric.

In terms of percieved usuability and quality of design, LLaMA 2 won handily, with its almost perfect score of 0.95 being 29 percent higher 
than worksheet #3's score in the normalized mean. Respondents said that it was "tidy" and "neat". One respondent commented that the LLaMA 2 
output was
more of a lesson plan than a worksheet, laying out instuctions and resources for the teacher as well. Some respondents really disliked the 
fact that the Command Nightly output supplied incorrect answers to questions. This is understandable, as correct answer
keys are a basic expectation most teachers have of their resources. It is also worth noting that the Command Nightly output's SD was almost 
twice as high as LLaMA 2's, showing that worksheet #3 elicited a very wide variance of opinion among respondents.

In summary, one can see that the LLaMA 2 output (worksheet #1) was considered superior by the majority of respondents, with little variance 
in opinion. While the Command-Nightly output (worksheet #3) did manage to beat LLaMA 2 in the engagement and interest
dimension, the output was highly controverial. Ultimately, it seems the egregiousness of worksheet #3's incorrect answer keys overshadowed its 
redeeming aspects.

# Error Analysis and Proposal for Future Improvements

While I can't compare any errors to held out data as suggested by the rubric, I would like to use this as an oppurtunity to analyze the 
experimental design of this project. One possible hole in this experiment is that respondents were not shown the outputs in a controlled
environment. Most were shown the outputs remotely via email or instant messaging services, and their responses were sent back similarly. 
As such, the studies author has no insight into what was happening around the respondents. Annoyances and stressors in the
environment could effect respondents' moods, and thus act as a confounding variable when assigning Likert scale scores. 
If this study were to be attempted again, this is one aspect of the experimental design that is low-hanging fruit for improvement.

Another thing that could be considered is whether or not respondents should be told that the outputs are AI generated. In this study, 
respondents were not told who or what made the worksheets shown to them. But is that the best way to do it? Would
knowing the origin of the worksheets effect how respondents feel about them? Perhaps one group should be shown the outputs with full 
knowledge that they are AI generated, while another should be shown the same outputs not knowing they are AI generated, as a control.
Only then could one establish if this a significant factor or not.

Additionally, the worksheets generated by the models in this experiment lacked somthing important- graphic
design. Most teachers would consider things like fonts, borders, clip art, etc. to be essential
when making a worksheet. These models simply aren't capable of doing that. Perhaps an ensemble approach
could be taken, with an image-based model like Stable Diffusion making the art for the worksheet,
and a text-based model making the actual content. Harmonizing these two tools into one may be challenging,
and significant effort would be required.

Finally, to address the elephant in the room, BLOOM's output in this experiment is simply unacceptable for
use in a classroom setting. It is incoherent, and frankly, bizarre. Going forward, one must seriously 
question if BLOOM is suitable at all for this application. Is BLOOM's poor performance attrivutable to 
it's architecture? Is it the dataset it was trained on? Or could it be an issue with hyperparameters?
Further inquiry is needed to get to the bottom of this

# Final Notes
Any code, prompts, or outputs associated with this project may be found
here: https://github.com/uazhlt-ms-program/ling-582-course-project-code-team-5 . I used an online GUI to 
instead of a Jupyter notebook or a Python script to run the various models, so there is no code associated with that.
However, the data, outputs, questionaire, and prompts from this experiemnt are freely available in the Github repo in the link.