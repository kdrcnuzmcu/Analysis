#region GÖREV 1

x = 8

y = 3.2

z = 8j + 18

a = "Hello World"

b = True

c = 23 < 22

l = [1, 2, 3, 4]

d = {
    "Name": "Jake",
    "Age": 27,
    "Address": "Downtown"
}

t = ("Machine Learning", "Data Science")

s = {"Python", "Machine Learning", "Data Science"}

for var in [x, y, z, a, b, c, l, d, t, s]:
    print(type(var))
# endregion

# region GÖREV 2
text = "The goal is to turn data into information, and information into insight."
text.upper()
text = text.split()
text
# endregion

# region GÖREV 3
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adim 1
len(lst)
# Adim 2
lst[0]
lst[10]
# Adim 3
data = lst[0:4]
# Adim 4
lst.pop(8)
# Adim 5
lst.insert(8, "N")
# endregion

# region GÖREV 4
dict = {"Cristian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]
        }
# Adim 1
dict.keys()
# Adim 2
dict.values()
# Adim 3
dict["Daisy"][1] = 13
# Adim 4
dict["Ahmet"] = ["Turkey", 24]
# Adim 5
dict.pop("Antonio")
# endregion

# region GÖREV 5
l = [2, 13, 18, 93, 22]
def func(nums):
    tek, cift = [], []
    for eleman in nums:
        if eleman % 2 == 0:
            cift.append(eleman)
        else:
            tek.append(eleman)
    return tek, cift

tek, cift = func(l)
# endregion

# region GÖREV 6
ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
for sira, ogrenci in enumerate(ogrenciler):
    if sira < 3:
        print(f"Mühendislik Fakültesi {sira + 1}. öğrenci {ogrenci}")
    else:
        print(f"Tıp Fakültesi {sira - 2}. öğrenci {ogrenci}")
# endregion

# region GÖREV 7
ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

zipList = tuple(zip(ders_kodu, kredi, kontenjan))
for ders in zipList:
    print(f"Kredisi {ders[1]} olan {ders[0]} kodlu dersin kontenjanı {ders[2]} kişidir.")
# endregion

# region GÖREV 8
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

kume1.issuperset(kume2)
kume2.issuperset(kume1)

kume1.issubset(kume2)
kume2.issubset(kume1)

kume1.intersection(kume2)
kume2.intersection(kume1)

kume1.difference(kume2)
kume2.difference(kume1)
# endregion