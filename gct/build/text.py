# Copyright 2026 HooRii Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Name normalisation and word overlap for English and Chinese text."""
import re

CJK = "一-鿿"
STOP = set("""the a an and or of to in on at for with from by is are was were be been it its this that these those i you he she they we
my your his her their our me him them as do did does have has had not no yes so very just also now currently still""".split())
_ZH_PRON = re.compile(r"^(我们|你们|他们|她们|咱们|我的|你的|他的|她的|我|你|他|她|咱|自己)")

# roles of which a person has only one at a time (partner / boyfriend / 对象 are left out: "partner" is also a business partner)
UNIQUE_ROLES = {"mother": "mother", "mom": "mother", "mum": "mother", "father": "father", "dad": "father", "wife": "spouse",
                "husband": "spouse", "spouse": "spouse",
                "妈妈": "mother", "母亲": "mother", "妈": "mother", "爸爸": "father", "父亲": "father", "爸": "father",
                "妻子": "spouse", "老婆": "spouse", "太太": "spouse", "丈夫": "spouse", "老公": "spouse", "爱人": "spouse"}


GENERIC = set(UNIQUE_ROLES) | {"partner", "boyfriend", "girlfriend", "fiance", "fiancé", "fiancee", "fiancée", "男朋友", "女朋友",
                                "对象", "伴侣", "未婚夫", "未婚妻", "mama", "papa", "mommy", "daddy", "grandma", "grandpa", "granny", "grandmother", "grandfather", "nana",
                                "sister", "brother", "sis", "bro", "son", "daughter", "aunt", "uncle", "cousin", "boss", "doctor",
                                "teacher", "baby", "kids", "the kids", "honey", "babe",
                                "奶奶", "爷爷", "外婆", "外公", "姥姥", "姥爷", "姐姐", "哥哥", "妹妹", "弟弟", "儿子", "女儿", "老板",
                                "老师", "医生", "宝宝", "孩子", "coach", "principal", "manager", "landlord", "nurse", "dentist",
                                "plumber", "vet", "nanny", "babysitter", "neighbor", "neighbour", "colleague", "roommate",
                                "班主任", "主任", "校长", "阿姨", "保姆", "师傅", "司机", "同事", "领导", "邻居", "室友", "同学", "物业"}


PRONOUNS = {"he", "she", "it", "they", "them", "him", "her", "his", "hers", "its", "their", "theirs", "we", "us", "i", "me", "you",
            "this", "that", "these", "those", "he or she", "他", "她", "它", "他们", "她们", "它们", "我", "你", "您", "我们", "你们",
            "这个", "那个"}


def keep_alias(a):
    """An alias worth keeping: not empty and not a pronoun."""
    return bool(str(a).strip()) and norm(a) not in PRONOUNS and str(a).strip().lower() not in PRONOUNS


def norm(s):
    s = re.sub(r"[\"'`’]s\b", "s", str(s).lower())
    s = re.sub(rf"[^a-z0-9{CJK}]+", " ", s)
    s = re.sub(r"^(the|a|an|my|our|his|her|their) ", "", s.strip())
    s = _ZH_PRON.sub("", s.strip())
    return s.strip()


def is_proper(s):
    """A name that identifies one thing by itself (capitalised word, digit or a Chinese name), not a description such as
    "User's dog", "my mom" or 我妈 / 周宁的车."""
    s = str(s).strip()
    if not s or re.search(r"(^|\s)(my|our|his|her|their|the)\s", s.lower() + " ") or re.search(r"['’]s\b", s):
        return False
    if norm(s) in GENERIC or not keep_alias(s):              # "Dad", "Mom", 妈妈, "He": not a name
        return False
    if re.search(rf"[{CJK}]", s):
        return not (_ZH_PRON.match(s) or "的" in s)
    return bool(re.search(r"[A-Z0-9]", s))


def tokens(text):
    """Content words: English words longer than one letter (minus stop words) and Chinese character bigrams."""
    t = str(text).lower()
    out = {w for w in re.findall(r"[a-z0-9]+", t) if w not in STOP and len(w) > 1}
    for run in re.findall(rf"[{CJK}]+", t):
        out |= {run[i:i + 2] for i in range(len(run) - 1)} if len(run) > 1 else {run}
    return out


def overlap(a, b):
    a, b = tokens(a), tokens(b)
    return len(a & b) / max(1, len(a | b))


ROLE_GROUPS = {
    "family": r"mother|father|mom|mum|dad|parent|sister|brother|sibling|son|daughter|child|kid|grand|aunt|uncle|cousin|niece|"
              r"nephew|in-law|妈|爸|父|母|姐|妹|哥|弟|儿|女儿|爷|奶|外公|外婆|姥|舅|姨|叔|伯|姑|婶|嫂|侄|孙|公公|婆婆|岳|亲家",
    "partner": r"partner|girlfriend|boyfriend|fianc|wife|husband|spouse|男朋友|女朋友|对象|伴侣|未婚|丈夫|妻|老公|老婆|爱人|太太",
    "work": r"colleague|coworker|co-worker|manager|boss|director|lead|coordinator|recruiter|collaborator|teammate|team|report|"
            r"employee|employer|client|customer|intern|engineer|designer|researcher|analyst|同事|领导|老板|经理|主管|下属|客户|合作",
    "service": r"plumber|electrician|mechanic|contractor|handyman|doctor|dentist|physician|pediatrician|nurse|therapist|physio|"
               r"vet|coach|instructor|trainer|tutor|teacher|principal|nanny|babysitter|cleaner|landlord|agent|driver|stylist|"
               r"barber|pharmacist|lawyer|accountant|医生|护士|老师|教练|阿姨|保姆|师傅|房东|中介|司机|律师|会计|物业|快递",
    "friend": r"friend|neighbo|roommate|classmate|朋友|邻居|室友|同学",
}


PERSONAL = {"friend", "partner"}


def roles_compatible(a, b):
    """Can two sets of role groups belong to one person? Yes when either is unknown, when they share a group, or when both are
    a friend or a partner (the friend who is later called a partner). A colleague and a plumber cannot, nor a brother and a
    friend."""
    if not a or not b:
        return True
    return bool(set(a) & set(b)) or (set(a) <= PERSONAL and set(b) <= PERSONAL)


def role_groups(role):
    """Coarse kinds of a free-text role ("plumber" -> {"service"}, "manager" -> {"work"}); empty when unknown."""
    r = str(role or "").lower()
    return {g for g, pat in ROLE_GROUPS.items() if re.search(pat, r)}


TITLE = re.compile(rf"^[{CJK}](?:老师|医生|大夫|阿姨|师傅|叔叔|主任|校长|经理|总|教练|护士|律师)$")


def is_title(name):
    """A surname with a title (王老师, 李医生): it names different people in different families, so it is resolved together with
    whose teacher / doctor it is."""
    return bool(TITLE.match(norm(name)))


def numbers(text):
    """Numbers and number-like codes in a text ("7:30" -> {"7", "30"}, "Apt 4B" -> {"4b"})."""
    return {(m.group(1) or m.group(0)).lower() for m in re.finditer(r"(\d+)(?:st|nd|rd|th)\b|\d+[a-zA-Z]?", str(text))}


def strip_names(text, names=()):
    """The text without the given names (people's names, the subject's name), so that word overlap compares what is said
    about them, not who it is about. Latin names are removed word by word (with a possessive 's), Chinese names as substrings."""
    t = str(text)
    for n in sorted({str(x) for x in names if x}, key=len, reverse=True):
        if re.search(rf"[{CJK}]", n):
            t = t.replace(n, " ")
        else:
            for w in n.split():
                if len(w) > 1:
                    t = re.sub(rf"\b{re.escape(w)}(?:['’]s)?\b", " ", t, flags=re.I)
    return t


_FUNC_ZH = re.compile(r"目前|现在|一直|仍然|已经|正在|住在|住|在|是|的|了|家|还|仍|也|都|就|已|着|过")


def same_value(old, new, names=()):
    """Does the newer statement give the same value as the older one? Compared without the names of the people involved:
    False at once when both carry numbers and the newer one has a number the older one lacks. Otherwise True when the
    remaining words overlap by half, when the newer one says nothing the older one does not (a vaguer
    restatement: "住在杭州" after "住在杭州市西湖区文三路12号"), when both carry the same numbers (a time, a room, an address,
    a phone number), or when every capitalised name the newer one uses already appears in the older one ("works at Ashby"
    after "works at Ashby High")."""
    o, n = strip_names(old, names), strip_names(new, names)
    a, b = numbers(o), numbers(n)
    if a and b and not b <= a:                            # "6 years old" -> "7 years old", "7 pm" -> "8 pm", 3岁 -> 4岁: a new value
        return False
    to, tn = tokens(o), tokens(n)
    if to and tn and len(to & tn) / len(to | tn) >= 0.5:
        return True
    if tn and tn <= to:
        return True
    core = [c for c in re.split(r"[^\u4e00-\u9fff]+", _FUNC_ZH.sub(" ", n)) if len(c) >= 2]
    if core and all(c in o for c in core):                # 住在南京 after 家住南京市建邺区…: the Chinese content is contained
        return True
    if b and a == b:
        return True
    caps = lambda t: {w for w in re.findall(r"\b[A-Z][A-Za-z]+", str(t))} - {"The", "A", "An"}  # noqa: E731
    ca, cb = caps(o), caps(n)
    return bool(cb) and cb <= ca and b <= a
