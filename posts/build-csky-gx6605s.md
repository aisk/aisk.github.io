% Build and Modify Linux System Image for C-sky based Gx6605s Board

The Gx6605s is a very cheap development board (39 Chinese Yuan with free shipping), which has a C-sky instruction set based CPU. The support for Linux kernel, GCC, and even Greenlet, is upstreamed.

![Gx6605s development board](https://c-sky.github.io/images/gx6605s_0.jpg)

I have a Gx6605s board which I bought years ago, and I only tried to run it with C-sky's official system image. It's a Buildroot-based Linux system, and there is no package manager for it. I can't install more packages / software on it, thus it is not useful for me, so I quickly lost interest in it.

When I found it in a box after moving to another city, I thought maybe I could do something with it, so I decided to give it a try.

But sadly, the company behind the development of the dev board and instruction set has abandoned the development for the dev board and C-sky instructions, and has continued their work on RiscV.

They have a Buildroot fork on GitLab which can produce the root file system for the image, but the support for this board and C-sky instruction set was dropped some time ago. Now this repo can't produce a system image for Gx6605s.

I made a dig into the repo and found that there is a branch called [`master_backup`](https://gitlab.com/c-sky/buildroot/-/tree/master_bakup), which was once their development branch capable of building C-sky CPU based system images. The build processes were run on GitLab Pipeline. So, you can fork this repo, and then run the GitLab Pipeline on the web page to get the generated system image.

But unfortunately, there is a file that Gx6605s Buildroot depended on, hosted at https://github.com/c-sky/tools, which has been deleted and changed to a new unrelated repo. Luckily, I found a mirror site that has this repo, so you can just apply this patch to get the file:

```diff
diff --git a/package/csky-debug/csky-debug.mk b/package/csky-debug/csky-debug.mk
index 241755019e..a318247ebf 100644
--- a/package/csky-debug/csky-debug.mk
+++ b/package/csky-debug/csky-debug.mk
@@ -6,7 +6,7 @@
 
 CSKY_DEBUG_VERSION = V4.2.0-tmp-20170411
 CSKY_DEBUG_SOURCE = DebugServerConsole-linux-x86_64-$(CSKY_DEBUG_VERSION).tar.gz
-CSKY_DEBUG_SITE = https://github.com/c-sky/tools/raw/master
+CSKY_DEBUG_SITE = https://isrc.iscas.ac.cn/gitlab/mirrors/github.com/c-sky_tools/-/raw/master
 
 define HOST_CSKY_DEBUG_INSTALL_CMDS
        mkdir -p $(HOST_DIR)/csky-debug
```

With this, you can build a system image for the Gx6605s board, and it's a little newer than the one provided on the C-sky official site. However, I want to modify the system image by adding more packages, such as MicroPython and even a GCC compiler. Therefore, I need to build it locally.

Take a look at the [`.gitlab-ci.yml`](https://gitlab.com/c-sky/buildroot/-/blob/master_bakup/.gitlab-ci.yml). I believe it's using a Docker image called `maohan001/ubuntu-buildroot` as the building environment. Please note that since they dropped support for the C-sky instruction set, if you want to build the image in the future, please pull and backup this image. It's not Dockerfile-based, so we don't know how to reproduce it.

I've never used Buildroot before, but from the GitLab Pipeline's log and the `.gitlab-ci.yml`, I assume one should run `make {config-name}` inside the Docker image with the `buildroot` root path to create a configuration for the specified development board. Then, run `make` to download all necessary files and build the compiler, kernel, libc, and userland.

So, for the Gx6605s, we should use `$ make csky_610_gx6605s_4.9_uclibc_br_defconfig`, and then `$ make`. You will get a `usb.img.xz` in the `output/images` folder. This should be exactly what you got in the GitLab Pipeline.

And as the `Buildroot`'s official website says, now we can run `$ make menuconfig` to customize the build process, such as adding MicroPython to the system.
