package = "objrcg"
version = "0.1-0"

source = {
   url = "git://github.com/lim0606/objrcg",
   tag = "master"
}

description = {
   summary = "a package of useful tools for torch-based object-recognition",
   detailed = [[
A package of useful functions/layers used in torch-based object-recognition.
   ]],
   homepage = "https://github.com/lim0606/objrcg",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}
