{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, devenv }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ devenv.flakeModule ];

      systems = [ "x86_64-linux" "aarch64-linux" ];
      perSystem = { config, system, pkgs, ... }: {
        devenv.shells.default = rec {
          packages = with pkgs; [
						wgpu-utils
            xorg.libX11
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXi
            xorg.libxcb
            libxkbcommon
            vulkan-loader
            wayland
          ];

          languages.rust.enable = true;

          enterShell = ''
						export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${builtins.toString (pkgs.lib.makeLibraryPath packages)}";
					'';
        };
      };
    };
}
