---
title: "Data Factory: My Experiences Training Industrial Scale A.I."
excerpt: "<br/><img src='./images/tts-image.jpg'>"
collection: portfolio
---

I am proud to say that I was able to complete my internship hours for University of Arizona’s (UAZ) Master of Science in Human Language Technology (HLT) program while working full-time at Invisible Technologies. It has been an eye -opening experience that has allowed me to learn what industrial scale LLM training looks like from the ground floor. Reinforcement learning from human feedback (or “hand-training”) is a nuanced process that involves many different moving parts. While many of the subtleties of what I learned are proprietary, a very high-level overview of what’s involved in a generic sense can be given.
One of the biggest bottlenecks in the world of AI is training data. There is a vast amount of training data to be had on the internet, but sadly, it is finite. As LLMs become larger, they need more and more data to be trained on. When there is no more natural data to be had, data must then be manufactured. Creating clean, useful data to train the client’s LLM with was my primary responsibility as an Advanced AI Data Trainer. At first, my job was to generate prose examples to help train an LLM to perform better in general. Later, I was promoted to a project where I generated code examples to show the client’s LLM how to program in Python according to professionally accepted best practices. 

Working with the prose-based data was fun. I got to write creatively and that’s always entertaining. Professionally, the interesting part was seeing the myriad rubrics and guidelines used to ensure alignment with the client’s expectations. The way the process worked was the LLM would say something, it would get flagged, and then we would write a “better” version of the flagged response. The high level of subjectivity involved makes this difficult. In most cases, no two people will qualify a piece of prose the same way. So, observing the inherently subjective process of writing prose to correct an LLM be governed by a set of objective criteria was interesting. While disputes about what may or may not be appropriate for the LLM to say were not uncommon, I have to say that everyone was mostly on the same page, which was remarkable. I think a lot of what helped was regular trainings to help cement the client’s expectations into the trainer’s minds. Certain kinds of errors or irregularities in the LLMs output were also referred to with a common vocabulary everyone was taught to use, which helped communication enormously. The whole operation is a testament to the power of standardization.

Later, I was invited to work on another project to write Python coding examples. While it still has its challenges, working to create code examples is a more straightforward process than creating prose examples. Unlike the endless rabbit hole of subjectivity that prose inhabits, code already has well accepted professional standards and guidelines. So, in this way, we can see that good code to show an LLM as a training example is synonymous with good code in any other professional context. A test-based development approach should be adopted, making sure that the code runs or throws the appropriate exception in any given situation. This also means taking care to validate inputs and really thinking about possible edge cases. For Python in particular, this also means ensuring that the code is PEP8 compliant. Originality is also very important. Any code written to give to the model as a training example must come from the trainer’s own mind and must be suitably complex. So, nothing from Github is allowed, nor are implementations of common coding challenges (palindrome checkers, fizz-buzz, etc.) unless they have some original twist.
Indeed, one of the toughest parts of this job is the constant creative pressure. Throughput is also essential, so the expectation is that trainers produce a finished Python example once every hour and a half. After a while, that can take a real mental toll on a person. Luckily, I found that my training in natural language processing (NLP) techniques given to me by the amazing faculty who teach the HLT program at UAZ left me uniquely well equipped for this challenge. NLP is a niche field. So, if you have the passion and imagination, it’s not hard to dream up original NLP ideas that can be implemented in an hour and a half. I will show you a few of my favorite examples and explain how they relate to what I learned during my studies at UAZ.

One of the things we learned early in the HLT program was basic linear algebra. So one day, I decided to write a function to calculate cosine similarity. This operation builds upon other linear algebra operations like finding the dot product and finding the magnitude of a vector, so it was complex enough to be permissible. Since third party libraries aren’t allowed in our examples, implementing cosine similarity without numpy also added to the relative complexity of the task. Here is my implementation, with some tests bundled in:

```python
def find_magnitude(vector: list[float]) -> float:

    return (sum(x * x for x in vector)) ** 0.5


def find_cosine_similarity(
    vector1: list[float], vector2: list[float], pad: bool
) -> float:

    if not vector1 or not vector2:
        raise ValueError("Cannot compute cosine similarity of empty vector(s).")

    if sum(vector1) + sum(vector2) == 0:
        raise ValueError(
            "Both vectors populated only by value '0', unable to divide zero by zero."
        )

    if pad:
        length_difference = abs(len(vector1) - len(vector2))
        if length_difference > 0:
            if len(vector1) < len(vector2):
                vector1.extend([0] * length_difference)
            else:
                vector2.extend([0] * length_difference)

    elif len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    cosine_similarity = dot_product / (
        find_magnitude(vector1) * find_magnitude(vector2)
    )

    return round(cosine_similarity, 2)


# TEST
assert find_cosine_similarity([1, 2, 1], [1, 1, 1], False) == 0.94
# TEST END

# TEST
assert find_cosine_similarity([1], [1, 1], True) == 0.71
# TEST END

# TEST
assert find_cosine_similarity([1, 1], [1], True) == 0.71
# TEST END

# TEST
try:
    find_cosine_similarity([1, 1], [1], False)
    assert False
except ValueError as e:
    assert str(e) == "Vectors must have the same length."
# TEST END

# TEST
try:
    find_cosine_similarity([], [1, 1, 1], True)
    assert False
except ValueError as e:
    assert str(e) == "Cannot compute cosine similarity of empty vector(s)."
# TEST END

# TEST
try:
    find_cosine_similarity([0], [0], False)
    assert False
except ValueError as e:
    assert (
        str(e)
        == "Both vectors populated only by value '0', unable to divide zero by zero."
    )
# TEST END
```	

It’s a simple implementation that uses generator expressions to implement vectorized computation. I also adding padding logic to the function and enforced that the input vectors be the same length. This wasn’t wholly necessary, but I wanted to add some extra complexity to the function. While finishing this example was satisfying, I can’t say this was a whole lot of fun to write. However, this idea bore greater creative fruit later.

In improv theatre, there is a simple adage of “Yes, and…”. This basically means that any idea an improviser produces can be built upon incrementally. While this philosophy of creative generativity is essential to improv, it is also widely advantageous for many other pursuits. During my time as an AI Data Trainer, I would often try to “Yes, And” my code ideas. So, after writing the cosine similarity implementation above, I asked myself, “Yes, this code calculates the cosine similarity of two vectors and… what else?” The answer I landed on was, “This code calculates the cosine similarity of two vectors and takes Chinese character stroke decompositions as inputs.” More clearly, the goal of this next Python example is to use stroke decompositions of Chinese characters to create feature vectors, and then find the cosine similarity of those two feature vectors. Having spent many semesters in my undergrad studying Chinese and over 6 years living and working in China, Chinese has become my second language and is somewhat of a fascination of mine. Before explaining further, it’s important to understand a little bit about how the Chinese writing system works.

Basically, there are 33 basic constituents or “strokes” that can be used to build any given Chinese character. These strokes are written and combined in a specific order. When a character’s strokes are put into a list in order of how they are written this is called a “stroke decomposition”. Since each character is a unique combination of strokes and the lists are ordered, this means a character’s stroke decomposition can be looked at like a “signature”.  For example, the character 来 (lái) which means “come” breaks down to `["一", "丨", "八", "一", "丷"]` whereas 请 (qíng) meaning “please” breaks down to `["㇊", "丶", "龶", "冂", "二"]`. I decided that if these unique signatures could be ascribed numerical values, one might be able to do something useful with them. Mindful that I had only an hour and a half to implement this, I achieved this by creating a toy integer encoding table, like this:

```python
stroke_encodings = {
        "冖": 1,
        "丶": 2,
        "㇛": 3,
        "一": 4,
        "丿": 5,
        "丨": 6,
        "八": 7,
        "丷": 8,
        "龶": 9,
        "冂": 10,
        "二": 11,
        "㇊": 12,
    }
```

I wanted to implement a more complex encoding scheme (maybe one-hot encoding?) but I didn’t have time. Here is the first implementation of my idea along with some tests:

```python
def find_magnitude(vector: list[float]) -> float:

    return (sum(x * x for x in vector)) ** 0.5


def hanzi_similarity(
    decomposition1: dict[str, list[str]], decomposition2: dict[str, list[str]]
) -> float:

    stroke_encodings = {
        "冖": 1,
        "丶": 2,
        "㇛": 3,
        "一": 4,
        "丿": 5,
        "丨": 6,
        "八": 7,
        "丷": 8,
        "龶": 9,
        "冂": 10,
        "二": 11,
        "㇊": 12,
    }

    if not decomposition1 or not decomposition2:
        raise ValueError("Cannot perform operations on empty data entries.")

    stroke_vector1 = [
        stroke for sublist in decomposition1.values() for stroke in sublist
    ]
    stroke_vector2 = [
        stroke for sublist in decomposition2.values() for stroke in sublist
    ]

    for stroke in set((stroke_vector1 + stroke_vector2)):
        if stroke not in set(stroke_encodings.keys()):
            raise ValueError(f"Stroke '{stroke}' is not in the encodings table.")

    integer_vector1 = [stroke_encodings[stroke] for stroke in stroke_vector1]
    integer_vector2 = [stroke_encodings[stroke] for stroke in stroke_vector2]

    if not integer_vector1 or not integer_vector2:
        raise ValueError("Cannot compute cosine similarity of empty vector(s).")

    length_difference = abs(len(integer_vector1) - len(integer_vector2))
    if length_difference > 0:
        if len(integer_vector1) < len(integer_vector2):
            integer_vector1.extend([0] * length_difference)
        else:
            integer_vector2.extend([0] * length_difference)

    dot_product = sum(x * y for x, y in zip(integer_vector1, integer_vector2))
    cosine_similarity = dot_product / (
        find_magnitude(integer_vector1) * find_magnitude(integer_vector2)
    )

    return cosine_similarity


import math

# TEST
expected_result = 0.9320995887754049
result = hanzi_similarity(
    {"安": ["冖", "丶", "㇛", "一", "丿"]}, {"来": ["一", "丨", "八", "一", "丷"]}
)
assert math.isclose(result, expected_result)
# TEST_END

# TEST
expected_result = 0.6157666004701773
result = hanzi_similarity(
    {"请": ["㇊", "丶", "龶", "冂", "二"]}, {"青": ["龶", "冂", "二"]}
)
assert math.isclose(result, expected_result)
# TEST_END

# TEST
try:
    hanzi_similarity(
        {"请": ["㇊", "丶", "龶", "冂", "二"]}, {"青": ["龶", "冂", "二", "Q"]}
    )
    assert False
except ValueError as e:
    assert str(e) == "Stroke 'Q' is not in the encodings table."
# TEST_END

# TEST
try:
    hanzi_similarity(
        {"请": ["㇊", "丶", "龶", "冂", "二"]}, {"青": ["龶", "冂", "二", " "]}
    )
    assert False
except ValueError as e:
    assert str(e) == "Stroke ' ' is not in the encodings table."
# TEST_END


# TEST
try:
    hanzi_similarity({"请": ["㇊", "丶", "龶", "冂", "二"]}, {"青": []})
    assert False
except ValueError as e:
    assert str(e) == "Cannot compute cosine similarity of empty vector(s)."
# TEST_END

# TEST
try:
    hanzi_similarity({"青": []}, {"请": ["㇊", "丶", "龶", "冂", "二"]})
    assert False
except ValueError as e:
    assert str(e) == "Cannot compute cosine similarity of empty vector(s)."
# TEST_END

# TEST
try:
    hanzi_similarity({}, {})
    assert False
except ValueError as e:
    assert str(e) == "Cannot perform operations on empty data entries."
# TEST_END
```

My earlier cosine similarity implementation forms the basis for this code, in addition to logic for parsing the inputs into integer vectors. The crudeness of the encoding scheme makes me wonder how meaningful the outputs of this function are, but it is good proof of concept if nothing else.

I knew that this idea could be fleshed out a lot more, so I said “Yes, And” to the idea once more and landed on the idea of possibly using this as a language teaching tool. There can be a significant difference between the traditional and simplified versions of Chinese characters. Having a way to quantify approximately how different a simplified character is from its traditional counterpart could be useful information for language learners. This information could help learners prioritize which simplified to traditional pairings to study. It could even help native speakers who are transitioning between locales where one writing standard is promoted over the other. Knowing that I was now intrigued with this idea and wanted to revisit it more in the future, I went out of my way to make it modular and extensible. I knew for an application like this to be viable, it would need a database of stroke decompositions. I decided to go ahead and define what those data entries should look like by writing a class that took JSON strings as input in the following format:

```json
{
    "请": {
        "traditional": "請",
        "decomposition": {
            "simplified": ["㇊", "丶", "龶", "冂", "二"],
            "traditional": ["一", "丶", "二", "口", "龶", "冂", "二"]
        },
        "glosses": ["please", "invite", "treat"]
    }
}
```

The glosses aren’t used here, but they may be useful later if I decide to extend the functionality of the module somehow, so I added them in. I also made the deliberate decision to use the simplified character as the key for the Python dictionaries these JSON strings would later be read into. This is based on the presupposition that more people know simplified characters and want to learn traditional characters than vice versa. Lastly, I decided to write this example as a class since this project was becoming increasingly complex and could only be achieved by writing multiple functions. Having this as a class also makes it easier to go back and add functionality later.  Here is the full implementation of this idea, with tests again:

```python
import json


class HanziVectors:

    def __init__(self) -> None:

        self.vocabulary = {}

        self.stroke_encodings = {
            "丶": 1,
            "一": 2,
            "丨": 3,
            "丿": 4,
            "乛": 5,
            "乚": 6,
            "𠃍": 7,
            "𠃌": 8,
            "㇀": 9,
            "㇁": 10,
            "㇂": 11,
            "㇃": 12,
            "㇄": 13,
            "㇅": 14,
            "㇇": 15,
            "㇈": 16,
            "㇉": 17,
            "㇊": 18,
            "㇋": 19,
            "㇌": 20,
            "㇍": 21,
            "㇎": 22,
            "㇏": 23,
            "㇐": 24,
            "㇑": 25,
            "㇚": 26,
            "㇛": 27,
            "冖": 28,
            "冂": 29,
            "八": 30,
            "丷": 31,
            "龶": 32,
            "二": 33,
        }

    def _magnitude(self, vector: list[float]) -> float:

        return (sum(x * x for x in vector)) ** 0.5

    def _extract_vectors(self, character: str) -> list[list[str], list[str]]:
        result = {}
        if character in self.vocabulary:
            data = self.vocabulary[character]
            if "decomposition" in data:
                vector_pairs = {}
                if "simplified" in data["decomposition"]:
                    vector_pairs["simplified"] = [
                        self.stroke_encodings.get(stroke, 0)
                        for stroke in data["decomposition"]["simplified"]
                    ]
                if "traditional" in data["decomposition"]:
                    vector_pairs["traditional"] = [
                        self.stroke_encodings.get(stroke, 0)
                        for stroke in data["decomposition"]["traditional"]
                    ]
                result = [
                    list(vector_pairs["simplified"]),
                    list(vector_pairs["traditional"]),
                ]
        return result

    def add_vocab(self, json_string: str) -> None:
        try:

            new_entry = json.loads(json_string)
            for key, value in new_entry.items():
                if key in self.vocabulary:
                    pass
                self.vocabulary[key] = value

        except json.JSONDecodeError as e:

            print(f"Error adding vocabulary: {str(e)}")

    def find_similarity(self, hanzi: str) -> float:

        if hanzi not in set(self.vocabulary.keys()):
            raise ValueError("Hanzi not in vocabulary.")

        vectors = self._extract_vectors(hanzi)

        integer_vector1 = vectors[0]
        integer_vector2 = vectors[1]

        if not integer_vector1 or not integer_vector2:
            raise ValueError("Cannot perform operations on empty data entries.")

        length_difference = abs(len(integer_vector1) - len(integer_vector2))

        if length_difference > 0:
            if len(integer_vector1) < len(integer_vector2):
                integer_vector1.extend([0] * length_difference)
            else:
                integer_vector2.extend([0] * length_difference)

        dot_product = sum(x * y for x, y in zip(integer_vector1, integer_vector2))
        cosine_similarity = dot_product / (
            self._magnitude(integer_vector1) * self._magnitude(integer_vector2)
        )

        return round(cosine_similarity, 2)


hanzi_vectors = HanziVectors()

qing = """
{
    "请": {
        "traditional": "請",
        "decomposition": {
            "simplified": ["㇊", "丶", "龶", "冂", "二"],
            "traditional": ["一", "丶", "二", "口", "龶", "冂", "二"]
        },
        "glosses": ["please", "invite", "treat"]
    }
}
"""
lai = """
{
    "来": {
        "traditional": "來",
        "decomposition": {
            "simplified": ["一", "丨", "八", "一", "丷"],
            "traditional": ["一", "丨", "八", "从"]
        },
        "glosses": ["come", "arrive", "future"]
    }
}
"""
# TEST
hanzi_vectors.add_vocab(
    """
{
    "请": {
        "traditional": "請",
        "decomposition": {
            "simplified": ["㇊", "丶", "龶", "冂", "二"],
            "traditional": ["一", "丶", "二", "口", "龶", "冂", "二"]
        },
        "glosses": ["please", "invite", "treat"]
    }
}
"""
)
hanzi_vectors.add_vocab(
    """
{
    "来": {
        "traditional": "來",
        "decomposition": {
            "simplified": ["一", "丨", "八", "一", "丷"],
            "traditional": ["一", "丨", "八", "从"]
        },
        "glosses": ["come", "arrive", "future"]
    }
}
"""
)

assert hanzi_vectors.vocabulary == {
    "请": {
        "traditional": "請",
        "decomposition": {
            "simplified": ["㇊", "丶", "龶", "冂", "二"],
            "traditional": ["一", "丶", "二", "口", "龶", "冂", "二"],
        },
        "glosses": ["please", "invite", "treat"],
    },
    "来": {
        "traditional": "來",
        "decomposition": {
            "simplified": ["一", "丨", "八", "一", "丷"],
            "traditional": ["一", "丨", "八", "从"],
        },
        "glosses": ["come", "arrive", "future"],
    },
}
# END TEST

# TEST
assert hanzi_vectors.find_similarity("来") == 0.7
# END TEST

# TEST
assert hanzi_vectors.find_similarity("请") == 0.59
# END TEST

# TEST
try:
    hanzi_vectors.find_similarity("f")
    assert False
except ValueError as e:
    assert str(e) == "Hanzi not in vocabulary."
#  END TEST

# TEST
try:
    hanzi_vectors.add_vocab("{asgfd}")
except json.JSONDecodeError as e:
    assert (
        str(e)
        == "Error adding vocabulary: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    )
    assert False
# END TEST

# TEST
hanzi_vectors.add_vocab(
    """
{
    "高": {
        "traditional": "翠",
        "decomposition": {
            "simplified": [],
            "traditional": []
        },
        "glosses": ["please", "invite", "treat"]
    }
}
"""
)

try:
    hanzi_vectors.find_similarity("高")
    assert False
except ValueError as e:
    assert str(e) == "Cannot perform operations on empty data entries."
# END TEST
```

The core functionality of finding the cosine similarity of two feature vectors is still there in the `find_similarity `method, but I altered the method to only take a single simplified character as input and return the cosine similarity of it and its traditional analouge. I also wrote a method for adding new vocabulary items, which could be used iteratively to add multiple items from a database someday. I also wrote a private method called `_extract_vectors` to pull the integer vectors out of the complex data structure the class is designed to work with. Again, the encoding scheme needs to be refined somehow and the code needs to be optimized, but this is gradually getting closer and closer to being polished. I look forward to working on this project more when I have more free time to do experiments like this.

Another interesting thing we learned in the HLT program was how search indexes work and statistical concepts like tf-idf (the product of term frequency and inverse document frequency) work. Tf-idf is a foundational concept in NLP for understanding how search engines analyze web content. I enjoyed studying this for my coursework because it really helped demystified how Google works (though their proprietary methods are obviously much more complex).  I think that’s pretty cool, so I thought it would be fun to code up an example of a function that uses tf-idf to show the client’s LLM. Here’s what I came up with:

```python
from math import log
from collections import Counter


def tokenize(text: str) -> str:

    puncts = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    no_punct = "".join([char.lower() if char not in puncts else " " for char in text])

    return no_punct.split()


def find_tf(tokenized_document: list[str]) -> dict[str, float]:

    tf_values = Counter(tokenized_document)

    total_words = len(tokenized_document)

    return {word: count / total_words for word, count in tf_values.items()}


def find_df(tokenized_corpus: list[list[str]]) -> dict[str, int]:

    df = Counter()

    for document in tokenized_corpus:
        unique_terms = set(document)
        df.update(unique_terms)

    return dict(df)


def make_tf_idf_table(corpus: list[tuple[int, str]]) -> list[dict[str, float]]:

    if not corpus:
        raise ValueError("Cannot perform operations on an empty corpus.")

    for doc in corpus:
        if not doc:
            raise ValueError("Corpus contains empty element(s).")

    tokenized_corpus = [tokenize(doc[1]) for doc in corpus]

    tfs = [find_tf(doc) for doc in tokenized_corpus]
    dfs = find_df(tokenized_corpus)
    corpus_length = len(tokenized_corpus)

    idfs = {word: log(corpus_length / frequency) for word, frequency in dfs.items()}

    tf_idf_table = [
        {word: tf[word] * idfs[word] for word in doc}
        for doc, tf in zip(tokenized_corpus, tfs)
    ]

    return tf_idf_table


from math import isclose

corpus = [
    (0, 'Jane said,"Look,look. I see a big yellow ear. See the yellow ear go.'),
    (
        1,
        'Sally said, "I see it. I see the big yellow ear. I want to go away in it. I want to go away, away."',
    ),
    (
        2,
        'Dick said, "Look up, Sally. You can see something. It is red and yellow. It can go up, up, up. It can go away."',
    ),
]


# TEST
assert make_tf_idf_table(corpus) == [
    {
        "jane": 0.07324081924454065,
        "said": 0.0,
        "look": 0.05406201441442192,
        "i": 0.02703100720721096,
        "see": 0.0,
        "a": 0.07324081924454065,
        "big": 0.02703100720721096,
        "yellow": 0.0,
        "ear": 0.05406201441442192,
        "the": 0.02703100720721096,
        "go": 0.0,
    },
    {
        "sally": 0.016894379504506847,
        "said": 0.0,
        "i": 0.06757751801802739,
        "see": 0.0,
        "it": 0.033788759009013694,
        "the": 0.016894379504506847,
        "big": 0.016894379504506847,
        "yellow": 0.0,
        "ear": 0.016894379504506847,
        "want": 0.0915510240556758,
        "to": 0.0915510240556758,
        "go": 0.0,
        "away": 0.05068313851352055,
        "in": 0.0457755120278379,
    },
    {
        "dick": 0.0457755120278379,
        "said": 0.0,
        "look": 0.016894379504506847,
        "up": 0.1831020481113516,
        "sally": 0.016894379504506847,
        "you": 0.0457755120278379,
        "can": 0.13732653608351372,
        "see": 0.0,
        "something": 0.0457755120278379,
        "it": 0.05068313851352055,
        "is": 0.0457755120278379,
        "red": 0.0457755120278379,
        "and": 0.0457755120278379,
        "yellow": 0.0,
        "go": 0.0,
        "away": 0.016894379504506847,
    },
]
# END TEST

# TEST
expected_output = [
    {
        "jane": 0.07324081924454065,
        "said": 0.0,
        "look": 0.05406201441442192,
        "i": 0.02703100720721096,
        "see": 0.0,
        "a": 0.07324081924454065,
        "big": 0.02703100720721096,
        "yellow": 0.0,
        "ear": 0.05406201441442192,
        "the": 0.02703100720721096,
        "go": 0.0,
    },
    {
        "sally": 0.016894379504506847,
        "said": 0.0,
        "i": 0.06757751801802739,
        "see": 0.0,
        "it": 0.033788759009013694,
        "the": 0.016894379504506847,
        "big": 0.016894379504506847,
        "yellow": 0.0,
        "ear": 0.016894379504506847,
        "want": 0.0915510240556758,
        "to": 0.0915510240556758,
        "go": 0.0,
        "away": 0.05068313851352055,
        "in": 0.0457755120278379,
    },
    {
        "dick": 0.0457755120278379,
        "said": 0.0,
        "look": 0.016894379504506847,
        "up": 0.1831020481113516,
        "sally": 0.016894379504506847,
        "you": 0.0457755120278379,
        "can": 0.13732653608351372,
        "see": 0.0,
        "something": 0.0457755120278379,
        "it": 0.05068313851352055,
        "is": 0.0457755120278379,
        "red": 0.0457755120278379,
        "and": 0.0457755120278379,
        "yellow": 0.0,
        "go": 0.0,
        "away": 0.016894379504506847,
    },
]


actual_output = make_tf_idf_table(corpus)
for doc_actual, doc_expected in zip(actual_output, expected_output):
    for word, expected_value in doc_expected.items():
        actual_value = doc_actual.get(word, 0)

        assert isclose(actual_value, expected_value)
# END TEST

# TEST
try:
    make_tf_idf_table([])
    assert False
except ValueError as e:
    assert str(e) == "Cannot perform operations on an empty corpus."
# END TEST

# TEST
try:
    make_tf_idf_table([["Hello"], []])
    assert False
except ValueError as e:
    assert str(e) == "Corpus contains empty element(s)."
# END TEST
```

Since the intent behind this piece of code was to toy with the concept of tf-idf, I thought it would be neat to write a function that calculates the tf-idf of every word in a corpus and returns them in a Python dictionary. This required the use of multiple helper functions, so an object-oriented approach might have been better here. If I ever come back to this little project, one of the first things I’d do to refactor the above code would be turning it into a class.

Tokenizers are another fundamental NLP concept I was exposed to during my studies at UAZ. Tokenizers are complex pieces of software, and implementing a good tokenizer in an hour and a half is basically impossible. But, it is possible to write a toy tokenizer in that amount of time. Here is a Python function that tokenizes English copulas (“to be” verbs). While it’s functionality is somewhat narrow, I am proud that it accounts for (most) contractions, including negative contractions. I didn’t have time to implement something that effectively distinguishes between an apostrophe “s” being used as a contraction for “is” and apostrophe “s” being used as a possessive, but the copula tokenizer covers most other edge cases I could think of at the time. The regexes aren’t clever, but they get the job done. Here is the code:

```python
from collections import Counter
import re


def be_counter(sentence: str) -> list[(list[str], list[str]), Counter]:

    if sentence == "":
        raise ValueError("Cannot tokenize an empty string!")

    contraction_patterns = {
        re.compile(r"(who's|isn't|he's|she's|it's)"): "is",
        re.compile(r"(you're|they're|we're|aren't)"): "are",
        re.compile(r"(I'm)"): "am",
        re.compile(r"(wasn't)"): "was",
        re.compile(r"(weren't)"): "were",
    }

    copulas = {
        "am": "VBP",
        "is": "VBZ",
        "are": "VBP",
        "was": "VBD",
        "were": "VBD",
        "be": "VB",
        "being": "VBG",
        "been": "VBG",
    }

    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s']", "", sentence)
    sentence = sentence.split(" ")

    for i in range(len(sentence)):
        for pattern, replacement in contraction_patterns.items():
            if pattern.match(sentence[i]):
                sentence[i] = replacement

    just_copulas = [token for token in sentence if token in copulas.keys()]
    tags = [copulas[token] if token in copulas.keys() else "???" for token in sentence]
    counts = Counter(just_copulas + tags)

    return [(sentence, tags), counts]


# TEST
assert be_counter("Isn't") == [(["is"], ["VBZ"]), Counter({"is": 1, "VBZ": 1})]
# TEST

assert be_counter("Isn't, WAS") == [
    (["is", "was"], ["VBZ", "VBD"]),
    Counter({"is": 1, "was": 1, "VBZ": 1, "VBD": 1}),
]
# TEST

assert be_counter("My name is Jeff.") == [
    (["my", "name", "is", "jeff"], ["???", "???", "VBZ", "???"]),
    Counter({"???": 3, "is": 1, "VBZ": 1}),
]
# TEST

try:
    be_counter("")
except ValueError as e:
    assert str(e) == "Cannot tokenize an empty string!"
# TEST
```
The function uses Penn Treebank  tags to tokenize copulas. I also made use of a counter object to count instances of each tag. The was no practical motive behind this design feature, I just thought it would be fun to do. 
I even wrote a toy Latin verb conjugator. Latin was the first foreign language I ever learned, and studying the subject is what started my interest in Linguistics, and later NLP. Latin (and realy any other ancient Indo-European language) has a notoriously complex verb conjugation system. So, writing code to automate the process seemed like low hanging fruit when I was looking for ideas to code. Here it is:

```python
def conjugator(verb: str, person: int, singular: bool) -> str:

    if not verb:
        raise ValueError("Value 'verb' must not be empty.")

    verb = verb.lower()

    if verb[-3:] != "are" or len(verb) <= 3:
        raise ValueError("Input must be a 1st conjugation infinitive verb.")

    if person not in {1, 2, 3}:
        raise ValueError(f"'{person}' not a valid person number.")

    conj_table = {
        "singular": {1: "o", 2: "as", 3: "at"},
        "plural": {1: "amus", 2: "atis", 3: "ant"},
    }

    stem = verb[:-3]

    if singular:
        new_ending = conj_table["singular"][person]
    else:
        new_ending = conj_table["plural"][person]

    return stem + new_ending


# TEST
assert conjugator("amare", 1, True) == "amo"
# END TEST

# TEST
assert conjugator("amare", 2, True) == "amas"
# END TEST


# TEST
assert conjugator("amare", 3, True) == "amat"
# END TEST


# TEST
assert conjugator("amare", 1, False) == "amamus"
# END TEST


# TEST
assert conjugator("amare", 2, False) == "amatis"
# END TEST


# TEST
assert conjugator("amare", 3, False) == "amant"
# END TEST

# TEST
try:
    conjugator("amare", 0, False)
    assert False
except ValueError as e:
    pass
# END TEST

# TEST
try:
    conjugator("are", 3, False)
    assert False
except ValueError as e:
    pass
# END TEST

# TEST
try:
    conjugator("habere", 3, False)
    assert False
except ValueError as e:
    pass
# END TEST

# TEST
try:
    conjugator("", 3, False)
    assert False
except:
    pass
# END_TEST
```

This code is just barely complex enough to be submittable, and I knew a regex-based implementation would make more sense. Having had lots of practice with regexes in the HLT program, I engineered another toy conjugator. This one uses regular expressions to do more of the heavy lifting as opposed to manual string manipulation. It only works for the first conjugation indicative present (the most regular one) like the first implementation, but this one also accounts for passive voice.

```python
import re


def regex_conjugator(verb: str, person: int, number: str, voice: str) -> str:

    if not verb:
        raise ValueError("Cannot compute empty string value.")

    first_conj_end_validation = re.compile(r"([a-z]+are)")

    verb = verb.lower()

    if not re.search(first_conj_end_validation, verb):
        raise ValueError("Input must be a 1st conjugation infinitive verb.")

    vocabs = {
        "person_vocab": {1, 2, 3},
        "number_vocab": {"plural", "singular"},
        "voice_vocab": {"active", "passive"},
    }

    if person not in vocabs["person_vocab"]:
        raise ValueError(f"Valid inputs for 'person' are  '1', '2', or '3'.")
    elif number not in vocabs["number_vocab"]:
        raise ValueError("Valid inputs for 'number' are 'singular' or 'plural'.")
    elif voice not in vocabs["voice_vocab"]:
        raise ValueError("Valid inputs for 'voice' are 'active' or 'passive'.")

    conj_table = {
        "active": {
            "singular": {1: "o", 2: "as", 3: "at"},
            "plural": {1: "amus", 2: "atis", 3: "ant"},
        },
        "passive": {
            "singular": {1: "abar", 2: "abaris", 3: "abatur"},
            "plural": {1: "abamur", 2: "abamini", 3: "abantur"},
        },
    }

    return re.sub(r"are", conj_table[voice][number][person], verb)


# TEST
person_vocab = [1, 2, 3]
number_vocab = ["plural", "singular"]
voice_vocab = ["active", "passive"]

permutations = []
for person in person_vocab:
    for number in number_vocab:
        for voice in voice_vocab:

            permutations.append(regex_conjugator("amare", person, number, voice))

assert permutations == [
    "amamus",
    "amabamur",
    "amo",
    "amabar",
    "amatis",
    "amabamini",
    "amas",
    "amabaris",
    "amant",
    "amabantur",
    "amat",
    "amabatur",
]
# TEST END

# TEST
try:
    regex_conjugator("amare", 0, "active", "passive")
    assert False
except ValueError as e:
    assert str(e) == "Valid inputs for 'person' are  '1', '2', or '3'."
# END TEST

# TEST
try:
    regex_conjugator("are", 3, "active", "passive")
    assert False
except ValueError as e:
    assert str(e) == "Input must be a 1st conjugation infinitive verb."
# END TEST


# TEST
try:
    regex_conjugator("amare", 3, "jefgn", "passive")
    assert False
except ValueError as e:
    assert str(e) == "Valid inputs for 'number' are 'singular' or 'plural'."
# END TEST

# TEST
try:
    regex_conjugator("amare", 3, "singular", "jhfdsbg")
    assert False
except ValueError as e:
    assert str(e) == "Valid inputs for 'voice' are 'active' or 'passive'."
# END TEST

# TEST
try:
    regex_conjugator("", 3, "singular", "jhfdsbg")
    assert False
except ValueError as e:
    assert str(e) == "Cannot compute empty string value."
# END TEST
```

While these are just a few of my favorite NLP examples I got to write during my time as an AI data trainer, there were many more that I submitted. Like I mentioned, NLP is a niche specialty. Since the model had not really ever been exposed to novel NLP code from a trainer before, I like to think that I was able to significantly extend the model’s knowledge and functionality in this domain.

After a few weeks of writing code examples, I was tapped for promotion to the role of Quality Analyst (QA). Now instead of writing code myself, I spend my day reviewing and fixing code that other trainers write. Technical skills are still paramount for this role, since we QAs are responsible for enforcing professional standards on all code submissions. However, the role is multifaceted. Giving feedback to other people about their code requires social grace and acumen. Otherwise, people will not be receptive to your advice. In my role as a QA, I am often called upon to lead workshops and mentor new hires, which is my favorite part of the job. Having spent over half a decade as an educator, stepping into the role of mentor feels very natural for me. I take great joy in getting to know the trainers I interact with. I always feel a burst of pride when I grade Python example written by a trainer I mentored and seeing that they got a perfect score or implemented a piece of feedback I had given previously. As a QA, I also get to dip my toes into the world of management as well. When there is a problem with a process, we QAs are called upon by management to brainstorm and implement creative solutions, since we are most acquainted with the day to day running of the project. It’s an exciting, fulfilling role that I think my training in the UAZ HLT program left me uniquely well equipped for. Having dipped my proverbial toe into the world of industrial scale AI, I cannot wait to see where my career will take me next!

All code examples mentioned in this article (and lots of other examples that weren’t) can be found [here]( https://github.com/pbarrett520/nlp_scripts/tree/main)!
