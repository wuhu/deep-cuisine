from collections import Counter, OrderedDict
import os
import re
import string

from nltk.corpus import wordnet
import numpy as np


stopwords = None
with open("stopwords_en.txt", "r") as f:
    stopwords = [l.strip() for l in f]

nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
nope = ["baking", "cooking", "dry", "room", "whip", "savory", "split", "peel", "shortening", "small", "medium",
        "red", "green", "white", "inch", "dark", "mix", "miniature", "firm", "fine", "cut", "bell", "cup", "starter",
        "inch", "plain", "cake", "pie", "ground", "large", "powder", "black", "taste", "fat", "liquid", "italian",
        "sweet", "jar", "light", "recipe", "chinese", "size", "paste", "raw", "quick", "garnish", "part", "kidney",
        "leaf", "hearts", "round", "crust", "pieces", "seeds", "sauce", "dish", "times", "temp", "hours", "pound",
        "hour", "process", "time", "meal", "min", "food", "note", "pressure", "pack", "information", "minutes",
        "style", "remove", "master", "table", "psi", "protein", "smoke", "coloring", "bottle", "pizza", "english",
        "gluten", "sodium", "heat", "yellow"]


def load(filename):
    with open(filename, 'r') as f:
        s = f.read()

    return s


def split_into_recipes(s):
    return s.split("MMMMM\r\n")


def load_from_dir(dirname):
    recipes = []
    for filename in os.listdir(dirname):
        recipes += split_into_recipes(load("{}/{}".format(dirname, filename)))
    return [remove_mm(r) for r in recipes]


def remove_mm(s):
    result = []
    for l in s.splitlines():
        if not l.startswith("MMM"):
            result.append(l)
    return "\n".join(result)


def extract_ingredients(r):
    spl = r.split("\n \n")
    for i, k in enumerate(spl):
        if "Yield" in k:
            break
    return spl[i+1]


def extract_description(r):
    spl = r.split("\n \n")
    for i, k in enumerate(spl):
        if "Yield" in k:
            break
    return "\n".join(spl[i+2:])


def get_all_ingredients(r):
    l = []
    for rr in r:
        try:
            l.append(extract_ingredients(rr))
        except:
            pass
    return l


def get_all_ingredients_desc(r):
    l = []
    for rr in r:
        try:
            l.append((extract_ingredients(rr), extract_description(rr)))
        except:
            pass
    return l


def split_quantity(line):
    q = line[:11]
    s = line[11:]
    return q.strip(), clean(s)


def split_quantities(q):
    lines = []
    for l in q.splitlines():
        if l[:11].strip():
            lines.append(l.lower())
        else:
            try:
                lines[-1] += l[10:].lower()
            except:
                pass
    return lines


def process_recipe(r):
    return [split_quantity(l) for l in split_quantities(r)]


def clean(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    s = regex.sub(' ', s)
    return " ".join(s.lower().split())


def counts(r):
    return Counter(" ".join(r).split(" "))


def most_common(s, c):
    best = 0
    bestw = None
    for word in s.split(" "):
        if word in stopwords:
            continue
        if word not in nouns:
            if word.endswith("s") and word[:-1] not in nouns:
                word = word[:-1]
            else:
                continue
        if word in nope:
            continue
        if word not in c:
            continue
        if c[word] > best:
            best = c[word]
            bestw = word
    return bestw


def filter_recipe(r, c):
    res = []
    for q, p in process_recipe(r):
        mc = most_common(p, c)
        if mc is not None:
            res.append(mc)
    return list(set(res))



def get_all_recipes():
    r = load_from_dir("recipes")
    return get_all_ingredients(r)

def get_counts(l):
    k = []
    for ll in l:
        try:
            k += process_recipe(ll)
        except:
            pass

    c = Counter(" ".join([p[1] for p in k]).split(" "))

    return Counter([most_common(t[1], c) for t in k])


def r2vec(r, c):
    l = filter_recipe(r, c)
    k = c.keys()
    v = np.zeros(len(c))
    for w in l:
        v[k.index(w)] = 1
    return v


def vec2r(v, c):
    r = []
    for i, k in zip(v, c):
        if i:
            r.append(k)
    return r
