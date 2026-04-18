"""
Massive Uzbek-first chat pair generator.

Produces 5000+ UNIQUE (user, assistant) pairs across many domains:
  - Greetings & smalltalk (variations)
  - Math (arithmetic, simple word problems)
  - Daily life Q&A
  - Tech/coding definitions
  - Instructions / how-to
  - Refusals
  - Time/date/personal questions
  - Follow-ups & confirmations

Used by dataset_gen.py to dramatically expand the training corpus.
"""

from __future__ import annotations
import random
from typing import List, Tuple

random.seed(123)

Pair = Tuple[str, str]

# ── Helpers ───────────────────────────────────────────

def _expand(templates: List[Tuple[str, str]], variants_u: List[List[str]] = None,
            variants_a: List[List[str]] = None) -> List[Pair]:
    """Expand templates with placeholder variants {0}, {1}..."""
    out: List[Pair] = []
    for ut, at in templates:
        out.append((ut, at))
    return out


# ── 1. GREETINGS — 200+ variations ────────────────────

GREET_BASE_UZ = [
    ("Salom", ["Salom!", "Salom, qanday yordam bera olaman?", "Salom! Yaxshimisiz?", "Salom, marhamat."]),
    ("salom", ["Salom!", "Salom! Bugun nima bilan yordam bera olaman?", "Salom, eshitaman."]),
    ("Assalomu alaykum", ["Vaalaykum assalom!", "Vaalaykum assalom, marhamat.", "Vaalaykum assalom, qanday yordam beray?"]),
    ("salom alaykum", ["Vaalaykum assalom!", "Vaalaykum assalom, eshitaman."]),
    ("Hayrli tong", ["Hayrli tong! Yaxshi kun bo'lsin.", "Hayrli tong, qanday yordam kerak?"]),
    ("Hayrli kun", ["Hayrli kun! Sizga qanday yordam beray?", "Hayrli kun, eshitaman."]),
    ("Hayrli kech", ["Hayrli kech! Qanday yordam beray?", "Hayrli kech, marhamat."]),
    ("Xayr", ["Xayr, omadli kun bo'lsin!", "Xayr, yaxshi qoling.", "Xayr! Yana keling."]),
    ("Yaxshi qoling", ["Yaxshi boring!", "Yaxshi qoling, omad tilayman."]),
    ("Ko'rishguncha", ["Ko'rishguncha!", "Ko'rishguncha, omadli bo'ling."]),
    ("Rahmat", ["Arzimaydi!", "Arzimaydi, marhamat.", "Yana savol bo'lsa, bemalol so'rang."]),
    ("Katta rahmat", ["Arzimaydi! Yana keling.", "Marhamat, har doim tayyorman."]),
    ("Tashakkur", ["Arzimaydi.", "Arzimaydi, har doim xizmatdaman."]),
    ("Yaxshimisan?", ["Rahmat, yaxshiman! Siz qandaysiz?", "Yaxshiman, o'zingiz-chi?"]),
    ("Qalaysan?", ["Yaxshi, rahmat! O'zingiz qalaysiz?", "Yaxshi, sizdan so'rashga ruxsat?"]),
    ("Ahvol qalay?", ["Yaxshi, rahmat. O'zingizniki?", "Hammasi joyida, sizdan so'rashga ruxsat?"]),
    ("Ishlar qalay?", ["Yaxshi, rahmat! Sizniki?", "Joyida, sizdan so'rashga ruxsat?"]),
    ("Ismingiz nima?", ["Men AI Coding Platform yordamchisiman.", "Mening ismim yo'q, men sun'iy intellektman."]),
    ("Sen kimsan?", ["Men sun'iy intellekt yordamchiman.", "Men AI Coding Platform yordamchisiman, kod va suhbatda yordam beraman."]),
    ("Sen nimasan?", ["Men sun'iy intellekt asosidagi yordamchiman.", "Men chat va kod bilan ishlovchi AI man."]),
    ("Kim seni yaratdi?", ["Meni AI Coding Platform jamoasi mahalliy o'qitgan.", "Men mahalliy ravishda noldan train qilingan modelman."]),
    ("Sen robotmisan?", ["Ha, men AI yordamchiman.", "Men sun'iy intellektman, robot deyish ham mumkin."]),
    ("Nima qila olasan?", ["Men o'zbek va ingliz tillarida suhbatlashaman, kod yozaman, savollarga javob beraman.", "Suhbat, kod yozish, tushuntirish — shularga yordam bera olaman."]),
    ("Yordam ber", ["Albatta! Qanday yordam kerak?", "Marhamat, savolingizni ayting."]),
    ("Yordam bera olasanmi?", ["Albatta, qanday savol?", "Ha, ayting marhamat."]),
    ("Tushundim", ["Yaxshi! Yana savolingiz bormi?", "Zo'r! Boshqa nima yordam beray?"]),
    ("Tushundim, rahmat", ["Arzimaydi! Yana keling.", "Marhamat, har doim xizmatdaman."]),
    ("Bo'pti", ["Yaxshi! Boshqa nima qilamiz?", "Mayli, yana nima yordam beray?"]),
    ("Xo'p", ["Yaxshi, davom etamiz.", "Bo'pti, eshitaman."]),
    ("Mayli", ["Yaxshi, davom etaylik.", "Bo'pti, eshitaman."]),
    ("Bilmadim", ["Hech qisi yo'q, men tushuntirib beraman.", "Mayli, birga tushunamiz."]),
    ("Eshityapsanmi?", ["Ha, eshityapman.", "Ha, marhamat ayting."]),
    ("Hozir bo'shmisan?", ["Ha, men har doim tayyorman.", "Albatta, eshitaman."]),
    ("Charchadingmi?", ["Yo'q, men charchamayman.", "Men AI man, charchamasdan ishlayman."]),
]

GREET_BASE_EN = [
    ("Hi", ["Hi!", "Hi, how can I help?", "Hi there!"]),
    ("Hello", ["Hello!", "Hello, what can I do for you?"]),
    ("Hey", ["Hey!", "Hey there, how can I help?"]),
    ("Good morning", ["Good morning! Hope you have a great day.", "Good morning, how can I help?"]),
    ("Good afternoon", ["Good afternoon! What can I do for you?", "Good afternoon!"]),
    ("Good evening", ["Good evening!", "Good evening, how can I help?"]),
    ("Bye", ["Goodbye! Have a great day.", "Bye, take care."]),
    ("Goodbye", ["Goodbye! See you next time.", "Goodbye, all the best."]),
    ("See you", ["See you!", "See you later."]),
    ("Thanks", ["You're welcome!", "Anytime!"]),
    ("Thank you", ["You're welcome!", "Glad I could help."]),
    ("How are you?", ["I'm doing well, thanks! How about you?", "Great, thanks for asking."]),
    ("What's up?", ["Not much, ready to help! What's up with you?", "Here to help, what do you need?"]),
    ("What's your name?", ["I'm the AI Coding Platform assistant.", "I don't have a personal name — I'm an AI assistant."]),
    ("Who are you?", ["I'm an AI assistant trained for chat and coding.", "I'm the AI Coding Platform assistant."]),
    ("Who made you?", ["I was trained by the AI Coding Platform team.", "I was built locally from scratch by the AI Coding Platform team."]),
    ("Are you a robot?", ["Yes, I'm an AI assistant.", "Yes, I'm an artificial intelligence."]),
    ("What can you do?", ["I can chat in Uzbek and English, write code, and answer questions.", "I help with chat, code, and explanations."]),
    ("Help me", ["Sure! What do you need help with?", "Of course, what's the question?"]),
    ("Got it", ["Great! Anything else?", "Awesome, what's next?"]),
    ("OK", ["Alright, what's next?", "Got it, ready when you are."]),
    ("Alright", ["Great, let's continue.", "Sounds good, what next?"]),
]

def gen_greetings() -> List[Pair]:
    out: List[Pair] = []
    for u, answers in GREET_BASE_UZ + GREET_BASE_EN:
        for a in answers:
            out.append((u, a))
    return out


# ── 2. MATH — arithmetic + word problems ──────────────

NUM_WORDS_UZ = ["bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz", "o'n"]

def gen_math_uz() -> List[Pair]:
    out: List[Pair] = []
    # Addition
    for a in range(0, 21):
        for b in range(0, 21):
            res = a + b
            for q in [
                f"{a}+{b}", f"{a} + {b}", f"{a} + {b} = ?",
                f"{a} ga {b} ni qo'shsak nechi bo'ladi?",
                f"{a} plyus {b}",
            ]:
                out.append((q, f"{a} + {b} = {res}"))
    # Subtraction (positive)
    for a in range(0, 21):
        for b in range(0, a + 1):
            res = a - b
            for q in [
                f"{a}-{b}", f"{a} - {b}", f"{a} - {b} = ?",
                f"{a} dan {b} ni ayirsak nechi bo'ladi?",
                f"{a} minus {b}",
            ]:
                out.append((q, f"{a} - {b} = {res}"))
    # Multiplication (small)
    for a in range(0, 13):
        for b in range(0, 13):
            res = a * b
            for q in [
                f"{a}*{b}", f"{a} * {b}", f"{a} x {b}",
                f"{a} ni {b} ga ko'paytirsak", f"{a} marta {b}",
            ]:
                out.append((q, f"{a} × {b} = {res}"))
    # Division (exact)
    for b in range(1, 11):
        for a_mult in range(0, 11):
            a = a_mult * b
            res = a_mult
            for q in [
                f"{a}/{b}", f"{a} / {b}", f"{a} ni {b} ga bo'lsak",
                f"{a} bo'linsin {b}",
            ]:
                out.append((q, f"{a} ÷ {b} = {res}"))
    # Word problems (small)
    word_uz = [
        ("Olmaning {a} tasi bor edi, yana {b} ta oldim. Jami nechta?",
         "{a} + {b} = {r} ta olma."),
        ("Do'konda {a} ta non bor edi, {b} tasi sotildi. Qancha qoldi?",
         "{a} - {b} = {r} ta non qoldi."),
        ("Bir savatda {a} ta tuxum bor. {b} ta savatda jami nechta tuxum?",
         "{a} × {b} = {r} ta tuxum."),
        ("{a} ta konfetni {b} bolaga teng bo'lib bersak, har biriga nechtadan tegadi?",
         "{a} ÷ {b} = {r} ta."),
    ]
    pairs_used = set()
    for tu, ta in word_uz:
        for a in [3, 5, 7, 10, 12, 15, 20]:
            for b in [2, 3, 4, 5]:
                if "/" in tu or "÷" in ta or "bo'lib" in tu:
                    if a % b != 0:
                        continue
                    r = a // b
                elif "qoldi" in tu:
                    if b > a:
                        continue
                    r = a - b
                elif "ko'pa" in tu or "savatda jami" in tu:
                    r = a * b
                else:
                    r = a + b
                q = tu.format(a=a, b=b)
                ans = ta.format(a=a, b=b, r=r)
                key = (q, ans)
                if key in pairs_used:
                    continue
                pairs_used.add(key)
                out.append(key)
    # Math in English
    for a in range(0, 21):
        for b in range(0, 21):
            out.append((f"What is {a} + {b}?", f"{a} + {b} = {a+b}"))
            out.append((f"{a} plus {b}", f"{a} + {b} = {a+b}"))
    for a in range(0, 13):
        for b in range(0, 13):
            out.append((f"What is {a} times {b}?", f"{a} × {b} = {a*b}"))
    return out


# ── 3. DAILY LIFE Q&A ─────────────────────────────────

DAILY_UZ: List[Pair] = [
    ("Bugun havo qanday?", "Kechirasiz, men real vaqtdagi havo ma'lumotini bilmayman. Havo ilovalaridan ko'ring."),
    ("Soat necha?", "Kechirasiz, men hozirgi vaqtni bilmayman. Telefoningizdan qarang."),
    ("Bugun kun qanday?", "Kechirasiz, men sanani aniq bila olmayman."),
    ("Yil nechinchi?", "Kechirasiz, men hozirgi yilni real vaqtda bilmayman."),
    ("Charchadim", "Bir oz dam oling. Kichik tanaffus ham foydali bo'ladi."),
    ("Uyqum kelyapti", "Yaxshi uxlang. Yetarli uyqu sog'liq uchun muhim."),
    ("Och qoldim", "Yaxshi taom yeb oling. Suv ichishni ham unutmang."),
    ("Suv ichgim keldi", "Iliq suv ichish foydali. Kuniga 6-8 stakan tavsiya etiladi."),
    ("Boshim og'riyapti", "Bir oz dam oling, suv iching. Davom etsa, shifokorga murojaat qiling."),
    ("Kayfiyatim yomon", "Buni eshitganim uchun afsusdaman. Yaqin odam bilan gaplashish yengillik beradi."),
    ("Quvonchliman", "Buni eshitganimdan xursandman! Yaxshi yangiliklar bormi?"),
    ("G'amginman", "Tushunaman. Sevgan ishingiz bilan shug'ullanish yordam berishi mumkin."),
    ("Bo'shman", "Yaxshi! Yangi narsa o'rganish yoki kichik loyiha qilishni boshlasak?"),
    ("Bandman", "Tushunarli. Vaqtingiz bo'lganda qaytishingiz mumkin."),
    ("Charchatdi", "Bir oz dam oling, keyin davom etaylik."),
    ("Sport bilan shug'ullansam bo'ladimi?", "Albatta! Muntazam sport sog'liq uchun juda foydali."),
    ("Kitob o'qishni tavsiya qilasizmi?", "Albatta, kitob o'qish bilim va so'z boyligini oshiradi."),
    ("Qanday qilib ko'proq o'rganish mumkin?", "Har kuni kichik miqdorda muntazam o'qish eng samarali."),
    ("Vaqtni qanday rejalashtiraman?", "To'-do ro'yxat tuzing va eng muhim ishlardan boshlang."),
    ("Qanday qilib ko'proq motivatsiya topaman?", "Aniq maqsad qo'ying va kichik g'alabalarni nishonlang."),
    ("Stress bilan qanday kurashaman?", "Chuqur nafas oling, sport bilan shug'ullaning, yetarli uxlang."),
    ("Yangi til o'rganmoqchiman", "Ajoyib! Har kuni 15-20 daqiqa muntazam mashq qilish foydali."),
    ("Sog'liqni qanday saqlash kerak?", "Yaxshi ovqatlanish, sport, yetarli uyqu va suv ichish — bular asos."),
    ("Pul tejashni o'rgating", "Daromad va xarajatlarni yozib boring, kerakmas xaridlardan saqlaning."),
    ("Eng yaxshi kasb qaysi?", "Sizga qiziq va yaxshi pul olib keladigan kasb eng yaxshi."),
    ("Ko'p ishlamoq kerakmi?", "Sifatli ishlash miqdordan muhimroq. Dam olishni ham unutmang."),
    ("Erta turishning foydasi bormi?", "Ha, ertalabki vaqt eng samarali soatlardan biri."),
    ("Suv ichishning foydasi nima?", "Suv organizmni tozalaydi, terini sog'lom saqlaydi va energiya beradi."),
]

DAILY_EN: List[Pair] = [
    ("How's the weather today?", "Sorry, I don't have real-time weather data. Check a weather app."),
    ("What time is it?", "Sorry, I don't know the current time. Check your device clock."),
    ("What's today's date?", "Sorry, I don't have access to the current date."),
    ("I'm tired", "Take a short break — even five minutes helps."),
    ("I'm sleepy", "Get some rest. Sleep is important for your health."),
    ("I'm hungry", "Have a good meal and drink some water."),
    ("I have a headache", "Rest a bit and drink water. If it persists, see a doctor."),
    ("I feel sad", "I'm sorry to hear that. Talking to someone close can help."),
    ("I'm happy", "Glad to hear it! Anything fun going on?"),
    ("I'm bored", "Want to learn something new or build a small project?"),
    ("How can I be more productive?", "Make a to-do list and start with the most important task."),
    ("How do I stay motivated?", "Set clear goals and celebrate small wins."),
    ("How do I deal with stress?", "Deep breathing, exercise, and good sleep all help a lot."),
    ("Should I read more?", "Yes, reading regularly improves knowledge and vocabulary."),
    ("How do I learn faster?", "Consistent daily practice beats long, infrequent sessions."),
    ("Is exercise important?", "Yes, regular exercise is great for both body and mind."),
    ("How can I save money?", "Track income and expenses, and avoid unnecessary purchases."),
    ("What's the best career?", "The one that interests you and pays your bills sustainably."),
    ("Should I work harder?", "Quality matters more than quantity. Don't forget to rest."),
]


# ── 4. TECH/CODING DEFINITIONS ────────────────────────

TECH_TERMS = [
    # (term_uz, term_en, def_uz, def_en)
    ("O'zgaruvchi", "variable",
     "O'zgaruvchi — bu qiymat saqlash uchun ishlatiladigan nomlangan joy. Masalan: `let x = 5`.",
     "A variable is a named storage for a value. Example: `let x = 5`."),
    ("Funksiya", "function",
     "Funksiya — bu ma'lum vazifani bajaradigan kod bo'lagi. Uni qayta-qayta chaqirish mumkin.",
     "A function is a reusable block of code that performs a task."),
    ("Massiv", "array",
     "Massiv — bu bir nechta qiymatni tartibli ro'yxatda saqlovchi tuzilma.",
     "An array is an ordered list that stores multiple values."),
    ("Obyekt", "object",
     "Obyekt — bu kalit-qiymat juftliklaridan iborat tuzilma. Masalan: `{ism: 'Ali'}`.",
     "An object is a structure of key-value pairs. Example: `{name: 'Ali'}`."),
    ("String", "string",
     "String — bu matn (so'z, jumla). Tirnoq ichida yoziladi: `'salom'`.",
     "A string is text written in quotes: `'hello'`."),
    ("Number", "number",
     "Number — bu son turi (5, 3.14, -2 kabi).",
     "Number is the numeric type (5, 3.14, -2)."),
    ("Boolean", "boolean",
     "Boolean — bu rost yoki yolg'on (true / false) qiymati.",
     "Boolean is true or false."),
    ("Loop", "loop",
     "Loop (sikl) — bu kodni bir necha marta takrorlash uchun ishlatiladi. `for`, `while`.",
     "A loop repeats code multiple times. `for` and `while` are common."),
    ("If-else", "if-else",
     "if-else — shartga qarab har xil kod bajarish uchun ishlatiladi.",
     "if-else runs different code depending on a condition."),
    ("Class", "class",
     "Class — obyektlar uchun shablon. OOP ning asosiy bloklaridan biri.",
     "A class is a blueprint for objects. Core to OOP."),
    ("API", "API",
     "API — dasturlar bir-biri bilan bog'lanish uchun ishlatadigan qoidalar to'plami.",
     "An API is a set of rules letting programs communicate."),
    ("HTTP", "HTTP",
     "HTTP — veb-sahifalarni uzatish uchun protokol. GET, POST kabi metodlari bor.",
     "HTTP is a protocol for transferring web pages. Methods include GET, POST."),
    ("JSON", "JSON",
     "JSON — ma'lumot almashish formati. Kalit-qiymat juftliklaridan iborat.",
     "JSON is a data exchange format made of key-value pairs."),
    ("Database", "database",
     "Database — ma'lumotlarni tartibli saqlash tizimi. Misollar: PostgreSQL, MongoDB.",
     "A database is an organized data storage system. Examples: PostgreSQL, MongoDB."),
    ("SQL", "SQL",
     "SQL — relyatsion bazadan ma'lumot olish va o'zgartirish tili.",
     "SQL is a language for querying relational databases."),
    ("Git", "Git",
     "Git — kod o'zgarishlarini kuzatuvchi versiya nazorat tizimi.",
     "Git is a version control system that tracks code changes."),
    ("GitHub", "GitHub",
     "GitHub — Git repositoriylar uchun bulutli platforma. Hamkorlik uchun ishlatiladi.",
     "GitHub is a cloud platform for Git repositories, used for collaboration."),
    ("HTML", "HTML",
     "HTML — veb-sahifaning tarkibini (matn, rasm, tugma) belgilaydigan markup tili.",
     "HTML is a markup language that defines the structure of a web page."),
    ("CSS", "CSS",
     "CSS — HTML elementlarining ko'rinishini (rang, o'lcham) belgilaydi.",
     "CSS controls the appearance (color, size) of HTML elements."),
    ("JavaScript", "JavaScript",
     "JavaScript — brauzerda va serverda (Node.js) ishlaydigan dasturlash tili.",
     "JavaScript is a programming language that runs in browsers and on Node.js."),
    ("TypeScript", "TypeScript",
     "TypeScript — JavaScript ning tip xavfsizligi qo'shilgan kengaytmasi.",
     "TypeScript is JavaScript with static type checking."),
    ("Python", "Python",
     "Python — soddaligi bilan mashhur ko'p maqsadli dasturlash tili.",
     "Python is a versatile programming language known for simplicity."),
    ("React", "React",
     "React — UI yaratish uchun komponent asosli JavaScript kutubxonasi.",
     "React is a component-based JavaScript library for building UIs."),
    ("Node.js", "Node.js",
     "Node.js — JavaScript ni server tomonida ishlatish uchun runtime.",
     "Node.js is a runtime that lets you run JavaScript on the server."),
    ("Async/await", "async/await",
     "async/await — Promise lar bilan ishlashning qulay sintaksisi. await Promise ni kutadi.",
     "async/await is convenient syntax for working with Promises. await pauses for the Promise."),
    ("Callback", "callback",
     "Callback — boshqa funksiyaga argument sifatida uzatiladigan funksiya.",
     "A callback is a function passed as an argument to another function."),
    ("Promise", "Promise",
     "Promise — kelajakda tugaydigan asinxron operatsiyani ifodalaydi.",
     "A Promise represents an async operation that completes in the future."),
    ("Frontend", "frontend",
     "Frontend — foydalanuvchi ko'radigan qism (UI). HTML, CSS, JS dan iborat.",
     "Frontend is what the user sees (UI). Built with HTML, CSS, JS."),
    ("Backend", "backend",
     "Backend — server tomonidagi mantiq, ma'lumot bilan ishlash, autentifikatsiya.",
     "Backend is the server-side logic, data handling, and authentication."),
    ("Framework", "framework",
     "Framework — dastur tuzilishini berib, kod yozishni tezlashtiradigan vosita.",
     "A framework provides a structure and speeds up development."),
    ("Library", "library",
     "Library — kerakli funksiyalar to'plami. Siz uni chaqirasiz, u sizni emas.",
     "A library is a collection of helpful functions. You call it, not vice versa."),
    ("Algoritm", "algorithm",
     "Algoritm — masalani yechish uchun aniq qadamlar ketma-ketligi.",
     "An algorithm is a clear sequence of steps to solve a problem."),
    ("Bug", "bug",
     "Bug — dasturdagi xato yoki nuqson.",
     "A bug is an error or flaw in a program."),
    ("Debug", "debug",
     "Debug — bug larni topib tuzatish jarayoni.",
     "Debugging is the process of finding and fixing bugs."),
    ("Cache", "cache",
     "Cache — tez-tez ishlatiladigan ma'lumotni vaqtinchalik saqlovchi joy.",
     "A cache is temporary storage for frequently used data."),
    ("Server", "server",
     "Server — boshqa kompyuterlar (klientlar) ga xizmat ko'rsatadigan kompyuter.",
     "A server is a computer that serves other computers (clients)."),
    ("Klient", "client",
     "Klient — serverdan xizmat so'rovchi dastur (masalan, brauzer).",
     "A client is a program that requests services from a server (e.g., browser)."),
    ("URL", "URL",
     "URL — internetdagi resursning manzili. Masalan: https://example.com.",
     "A URL is the address of a resource on the internet."),
    ("Domain", "domain",
     "Domain — sayt nomi (masalan: google.com).",
     "A domain is a website's name (e.g., google.com)."),
    ("DNS", "DNS",
     "DNS — domain nomini IP manziliga aylantiruvchi tizim.",
     "DNS translates domain names into IP addresses."),
    ("HTTPS", "HTTPS",
     "HTTPS — shifrlangan, xavfsiz HTTP ulanishi.",
     "HTTPS is encrypted, secure HTTP."),
    ("TCP", "TCP",
     "TCP — ishonchli ma'lumot uzatish protokoli.",
     "TCP is a reliable data transmission protocol."),
    ("IP manzil", "IP address",
     "IP manzil — har bir tarmoqdagi qurilmaning noyob raqami.",
     "An IP address is a unique number for every device on a network."),
    ("Linux", "Linux",
     "Linux — bepul, ochiq kodli operatsion tizim. Serverlarda keng qo'llaniladi.",
     "Linux is a free, open-source OS widely used on servers."),
    ("Docker", "Docker",
     "Docker — dasturlarni konteynerlarda ishga tushirish uchun vosita.",
     "Docker runs applications in lightweight containers."),
    ("REST", "REST",
     "REST — HTTP orqali ishlaydigan API arxitektura uslubi.",
     "REST is an API architecture style using HTTP."),
    ("GraphQL", "GraphQL",
     "GraphQL — kerakli ma'lumotni so'rab olish uchun zamonaviy API tili.",
     "GraphQL is a modern API query language for fetching exact data needs."),
    ("WebSocket", "WebSocket",
     "WebSocket — ikki tomonlama real-time ulanish protokoli.",
     "WebSocket is a protocol for two-way real-time communication."),
    ("Cookie", "cookie",
     "Cookie — brauzerda saqlanadigan kichik ma'lumot bo'lagi.",
     "A cookie is a small data piece stored in the browser."),
    ("LocalStorage", "localStorage",
     "localStorage — brauzerda doimiy saqlash uchun ishlatiladi.",
     "localStorage stores data in the browser persistently."),
    ("Token", "token",
     "Token — autentifikatsiya yoki avtorizatsiyada ishlatiladigan kod.",
     "A token is a code used for authentication or authorization."),
    ("OAuth", "OAuth",
     "OAuth — boshqa xizmat orqali kirish uchun standart protokol.",
     "OAuth is a standard protocol for third-party authentication."),
    ("Hashing", "hashing",
     "Hashing — matnni qaytmas usulda kodga aylantirish (parollar uchun).",
     "Hashing converts text into an irreversible code (used for passwords)."),
    ("Encryption", "encryption",
     "Encryption — ma'lumotni shifrlash. Faqat kalit bilan ochish mumkin.",
     "Encryption secures data so only the key can decrypt it."),
    ("Recursion", "recursion",
     "Rekursiya — funksiyaning o'zini chaqirishi.",
     "Recursion is when a function calls itself."),
    ("Stack", "stack",
     "Stack — LIFO (oxirgi kirgan birinchi chiqadi) tuzilma.",
     "A stack is a LIFO data structure (last in, first out)."),
    ("Queue", "queue",
     "Queue — FIFO (birinchi kirgan birinchi chiqadi) tuzilma.",
     "A queue is a FIFO data structure (first in, first out)."),
    ("Hash map", "hash map",
     "Hash map — kalit-qiymat juftliklarini tez izlash uchun tuzilma.",
     "A hash map stores key-value pairs for fast lookup."),
    ("Big O", "Big O",
     "Big O — algoritm samaradorligini tavsiflovchi belgi (vaqt yoki xotira).",
     "Big O describes how an algorithm scales (time or memory)."),
]

QUESTION_TEMPLATES_UZ = [
    "{} nima?",
    "{} nima degani?",
    "{} ni tushuntir",
    "{} haqida ayt",
    "{} qanday ishlaydi?",
    "{} nima uchun kerak?",
]

QUESTION_TEMPLATES_EN = [
    "What is {}?",
    "Define {}",
    "Explain {}",
    "Tell me about {}",
    "How does {} work?",
    "Why do we need {}?",
]

def gen_tech() -> List[Pair]:
    out: List[Pair] = []
    for term_uz, term_en, def_uz, def_en in TECH_TERMS:
        for tpl in QUESTION_TEMPLATES_UZ:
            out.append((tpl.format(term_uz), def_uz))
        for tpl in QUESTION_TEMPLATES_EN:
            out.append((tpl.format(term_en), def_en))
    return out


# ── 5. CODE SNIPPETS ──────────────────────────────────

CODE_SNIPPETS_UZ: List[Pair] = [
    ("Hello world dasturini JavaScript da yoz",
     "```javascript\nconsole.log('Salom, dunyo!');\n```"),
    ("Hello world Python da",
     "```python\nprint('Salom, dunyo!')\n```"),
    ("Ikki sonni qo'shadigan funksiya yoz",
     "```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```"),
    ("Ikki sonni ko'paytiruvchi funksiya",
     "```javascript\nfunction multiply(a, b) {\n  return a * b;\n}\n```"),
    ("Massiv elementlari yig'indisini topish",
     "```javascript\nconst sum = arr => arr.reduce((a, b) => a + b, 0);\n```"),
    ("Massivdagi eng katta sonni topish",
     "```javascript\nconst max = arr => Math.max(...arr);\n```"),
    ("String ni teskari yozish",
     "```javascript\nconst reverse = s => s.split('').reverse().join('');\n```"),
    ("Tub son ekanligini tekshirish",
     "```javascript\nfunction isPrime(n) {\n  if (n < 2) return false;\n  for (let i = 2; i * i <= n; i++)\n    if (n % i === 0) return false;\n  return true;\n}\n```"),
    ("Faktorial hisoblash",
     "```javascript\nfunction factorial(n) {\n  return n <= 1 ? 1 : n * factorial(n - 1);\n}\n```"),
    ("Fibonacci ketma-ketligi",
     "```javascript\nfunction fib(n) {\n  return n < 2 ? n : fib(n-1) + fib(n-2);\n}\n```"),
    ("API dan ma'lumot olish (fetch)",
     "```javascript\nconst data = await fetch('https://api.example.com').then(r => r.json());\n```"),
    ("Massivni saralash (sort)",
     "```javascript\nconst sorted = arr.sort((a, b) => a - b);\n```"),
    ("Obyektdan kalitlarni olish",
     "```javascript\nconst keys = Object.keys(obj);\n```"),
    ("Sondagi raqamlar yig'indisi",
     "```javascript\nfunction digitSum(n) {\n  return String(n).split('').reduce((s, d) => s + +d, 0);\n}\n```"),
    ("Qatordagi unli harflar soni",
     "```javascript\nconst vowels = s => (s.match(/[aeiouAEIOU]/g) || []).length;\n```"),
    ("Sonni harflarda yozish (1 dan 10 gacha)",
     "```javascript\nconst words = ['nol','bir','ikki','uch','to\\'rt','besh','olti','yetti','sakkiz','to\\'qqiz','o\\'n'];\nconst toWord = n => words[n] || String(n);\n```"),
    ("React da oddiy komponent",
     "```jsx\nfunction Hello({name}) {\n  return <h1>Salom, {name}!</h1>;\n}\n```"),
    ("React da useState",
     "```jsx\nconst [count, setCount] = useState(0);\n```"),
    ("Python da fayl o'qish",
     "```python\nwith open('file.txt') as f:\n    text = f.read()\n```"),
    ("Python da JSON o'qish",
     "```python\nimport json\nwith open('data.json') as f:\n    data = json.load(f)\n```"),
]

CODE_SNIPPETS_EN: List[Pair] = [
    ("Write hello world in JavaScript",
     "```javascript\nconsole.log('Hello, world!');\n```"),
    ("Hello world in Python",
     "```python\nprint('Hello, world!')\n```"),
    ("Function to add two numbers",
     "```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```"),
    ("Function to multiply two numbers",
     "```javascript\nfunction multiply(a, b) {\n  return a * b;\n}\n```"),
    ("Sum of array elements",
     "```javascript\nconst sum = arr => arr.reduce((a, b) => a + b, 0);\n```"),
    ("Find max in array",
     "```javascript\nconst max = arr => Math.max(...arr);\n```"),
    ("Reverse a string",
     "```javascript\nconst reverse = s => s.split('').reverse().join('');\n```"),
    ("Check if a number is prime",
     "```javascript\nfunction isPrime(n) {\n  if (n < 2) return false;\n  for (let i = 2; i * i <= n; i++)\n    if (n % i === 0) return false;\n  return true;\n}\n```"),
    ("Factorial",
     "```javascript\nfunction factorial(n) {\n  return n <= 1 ? 1 : n * factorial(n - 1);\n}\n```"),
    ("Fibonacci sequence",
     "```javascript\nfunction fib(n) {\n  return n < 2 ? n : fib(n-1) + fib(n-2);\n}\n```"),
    ("Fetch JSON from an API",
     "```javascript\nconst data = await fetch('https://api.example.com').then(r => r.json());\n```"),
    ("Sort an array",
     "```javascript\nconst sorted = arr.sort((a, b) => a - b);\n```"),
    ("Get keys of an object",
     "```javascript\nconst keys = Object.keys(obj);\n```"),
    ("Sum of digits in a number",
     "```javascript\nconst digitSum = n => String(n).split('').reduce((s, d) => s + +d, 0);\n```"),
    ("Count vowels in a string",
     "```javascript\nconst vowels = s => (s.match(/[aeiouAEIOU]/g) || []).length;\n```"),
    ("Read a file in Python",
     "```python\nwith open('file.txt') as f:\n    text = f.read()\n```"),
    ("Read JSON in Python",
     "```python\nimport json\nwith open('data.json') as f:\n    data = json.load(f)\n```"),
    ("Simple React component",
     "```jsx\nfunction Hello({name}) {\n  return <h1>Hello, {name}!</h1>;\n}\n```"),
    ("React useState example",
     "```jsx\nconst [count, setCount] = useState(0);\n```"),
]


# ── 6. INSTRUCTIONS / HOW-TO ──────────────────────────

HOWTO_UZ: List[Pair] = [
    ("Qanday qilib React loyihasi ochish mumkin?",
     "`npx create-react-app my-app` buyrug'ini ishga tushiring. Yangi papka avtomatik yaratiladi."),
    ("Node.js ni qanday o'rnataman?",
     "nodejs.org saytidan LTS versiyani yuklab oling va o'rnating. Keyin `node -v` bilan tekshiring."),
    ("Python ni qanday o'rnataman?",
     "python.org saytidan oxirgi versiyani yuklab oling. O'rnatishda 'Add to PATH' belgisini bosing."),
    ("VS Code ga qanday extension o'rnataman?",
     "Chap paneldagi Extensions belgisini bosing, kerakli kengaytmani izlang va Install bosing."),
    ("Git da yangi repo qanday yaratiladi?",
     "`git init` buyrug'ini papkada ishga tushiring. Keyin `git add` va `git commit` qiling."),
    ("Git da qanday push qilaman?",
     "`git push origin main` buyrug'ini ishlating. Avval remote o'rnatilgan bo'lsin."),
    ("Tarmoq xatosini qanday tuzataman?",
     "Avval internet ulanishini tekshiring, keyin URL to'g'riligini, oxiri server ishlayotganini."),
    ("npm package qanday o'rnataman?",
     "`npm install paket-nomi` buyrug'i bilan. Global uchun `-g` qo'shing."),
    ("HTML faylni qanday ochaman?",
     "Brauzerda File → Open File ni bosing yoki HTML faylni brauzerga sudrang."),
    ("CSS sinflarini qanday qo'shaman?",
     "HTML elementiga `class=\"sinfnomi\"` qo'shing va CSS da `.sinfnomi { ... }` yozing."),
]

HOWTO_EN: List[Pair] = [
    ("How do I create a React project?",
     "Run `npx create-react-app my-app`. A new folder will be created automatically."),
    ("How do I install Node.js?",
     "Download the LTS version from nodejs.org. Verify with `node -v`."),
    ("How do I install Python?",
     "Download from python.org. During install, check 'Add to PATH'."),
    ("How do I install a VS Code extension?",
     "Click the Extensions icon, search for the extension, and click Install."),
    ("How do I create a Git repository?",
     "Run `git init` in your folder, then use `git add` and `git commit`."),
    ("How do I push to Git?",
     "Use `git push origin main` after setting up the remote."),
    ("How do I install an npm package?",
     "Run `npm install package-name`. Add `-g` for global install."),
    ("How do I open an HTML file?",
     "Open it in a browser with File → Open File, or drag the HTML into the browser."),
    ("How do I add CSS classes?",
     "Add `class=\"name\"` to an HTML element and write `.name { ... }` in CSS."),
]


# ── 7. REFUSALS ───────────────────────────────────────

REFUSALS_UZ: List[Pair] = [
    ("Birovning parolini buz", "Kechirasiz, men bunga yordam bera olmayman. Bu xavfsizlikka zid."),
    ("Virus yozib ber", "Kechirasiz, zararli dastur yaratishda yordam bera olmayman."),
    ("Birovning hisobiga kir", "Kechirasiz, bunday so'rovga javob bera olmayman."),
    ("Saytni hack qil", "Kechirasiz, men noqonuniy ishlarda yordam bermayman."),
    ("Phishing sahifa yoz", "Kechirasiz, bu insonlarga zarar yetkazadi, men yordam bera olmayman."),
    ("Spam yuborish skript yoz", "Kechirasiz, men spam vositalarini yarata olmayman."),
    ("Birovni tahdid qilish xabarini yoz", "Kechirasiz, men bunga yordam bermayman."),
    ("Yolg'on ma'lumot tarqat", "Kechirasiz, men yolg'on ma'lumot yaratishda yordam bermayman."),
]

REFUSALS_EN: List[Pair] = [
    ("Hack someone's password", "Sorry, I can't help with that — it's against safety guidelines."),
    ("Write malware", "Sorry, I won't help create malicious software."),
    ("Break into an account", "Sorry, I can't assist with that request."),
    ("Hack a website", "Sorry, I don't help with illegal activities."),
    ("Write phishing page", "Sorry, that would harm people. I can't help."),
    ("Write spam script", "Sorry, I can't create spam tools."),
    ("Write a threat message", "Sorry, I won't help with that."),
    ("Spread misinformation", "Sorry, I won't help create false information."),
]


# ── PUBLIC API ────────────────────────────────────────

def all_pairs() -> List[Pair]:
    out: List[Pair] = []
    out.extend(gen_greetings())
    out.extend(gen_math_uz())
    out.extend(DAILY_UZ)
    out.extend(DAILY_EN)
    out.extend(gen_tech())
    out.extend(CODE_SNIPPETS_UZ)
    out.extend(CODE_SNIPPETS_EN)
    out.extend(HOWTO_UZ)
    out.extend(HOWTO_EN)
    out.extend(REFUSALS_UZ)
    out.extend(REFUSALS_EN)
    # de-dup
    seen = set()
    uniq: List[Pair] = []
    for u, a in out:
        key = (u.strip(), a.strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((u.strip(), a.strip()))
    return uniq


if __name__ == "__main__":
    pairs = all_pairs()
    print(f"Total unique pairs: {len(pairs)}")
    by_lang_uz = sum(1 for u, _ in pairs if any(c in u for c in "qoʻgʻchshyzxoʻ") or 'sh' in u.lower())
    print(f"Sample (first 5):")
    for u, a in pairs[:5]:
        print(f"  Q: {u}\n  A: {a[:80]}\n")
    print(f"Sample (last 5):")
    for u, a in pairs[-5:]:
        print(f"  Q: {u}\n  A: {a[:80]}\n")
