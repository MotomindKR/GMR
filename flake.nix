{
  description = "Reproducible GMR development and retargeting environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        runtimeLibraries = with pkgs; [
          libGL
          libglvnd
          glfw
          stdenv.cc.cc.lib
          libx11
          libxcursor
          libxext
          libxi
          libxinerama
          libxrandr
          libxcb
          zlib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ffmpeg
            git
            python311
            uv
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;
          UV_CACHE_DIR = ".uv-cache";
          UV_NO_PROJECT = "1";
          UV_PROJECT_ENVIRONMENT = ".venv";
          UV_PYTHON_DOWNLOADS = "never";

          shellHook = ''
            set -e

            lock_hash="$(sha256sum requirements.lock | cut -d' ' -f1)"
            stamp="$UV_PROJECT_ENVIRONMENT/.gmr-requirements-lock"

            if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] \
              || [ "$(cat "$stamp" 2>/dev/null)" != "$lock_hash" ] \
              || ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import grpc, mujoco" 2>/dev/null; then
              uv venv --python ${pkgs.python311}/bin/python --clear "$UV_PROJECT_ENVIRONMENT"
              uv pip sync \
                --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
                --torch-backend cpu \
                requirements.lock
              printf '%s\n' "$lock_hash" > "$stamp"
            fi

            source "$UV_PROJECT_ENVIRONMENT/bin/activate"

            if [ -z "''${MUJOCO_GL:-}" ]; then
              if [ -n "''${DISPLAY:-}" ]; then
                export MUJOCO_GL=glfw
              else
                export MUJOCO_GL=egl
              fi
            fi
          '';
        };
      }
    );
}
