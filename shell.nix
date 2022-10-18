{ pkgs ? import <nixpkgs> { } }:
with pkgs; mkShell rec {
  nativeBuildInputs = [
    pkgconfig
    llvmPackages.bintools # To use lld linker
  ];
  buildInputs = [
    udev
    alsaLib
    vulkan-loader
    xlibsWrapper
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi # To use x11 feature
    libxkbcommon
    wayland # To use wayland feature
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
}
