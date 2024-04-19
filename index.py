import os
import subprocess
from dataclasses import dataclass
from datetime import datetime


print("% Buzhanxian")


@dataclass
class Post():
    filename: str
    createtime: datetime


filenames = (x for x in os.listdir("posts") if x.endswith(".md"))
posts: list[Post] = []


for filename in filenames:
    cmd = f"git log --follow --format=%ad --date unix posts/{filename}"
    result = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    createtime = datetime.fromtimestamp(int(result.stdout.split()[-1]))
    filename = filename.replace(".md", ".html")
    posts.append(Post(filename=filename, createtime=createtime))


posts.sort(key=lambda x: x.createtime)
posts.reverse()


for post in posts:
    print(f"- [{post.filename}](posts/{post.filename}) &middot; *{post.createtime}*")
