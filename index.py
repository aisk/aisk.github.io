import os


print("% My Blog")

posts = (x for x in os.listdir("posts") if x.endswith(".md"))

for file in os.listdir("posts"):
    if not file.endswith(".html"):
        continue
    print(f"- [{file}](posts/{file})")

