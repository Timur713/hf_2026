#!/usr/bin/env bash
set -Eeuo pipefail

# Idempotent Ubuntu LTS bootstrap for the A10/A5000 inference servers.
# Required when dataset/ is absent:
#   DATASET_SOURCE=/path/to/archive-or-unpacked-dir
# or
#   DATASET_SOURCE=https://... DATASET_PASSWORD=...
# Password-protected HTTP sources may also set DATASET_USERNAME.

repo_dir="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
data_root="${DATA_ROOT:-${repo_dir}/dataset}"
venv_dir="${VENV_DIR:-${repo_dir}/.venv}"
install_driver="${INSTALL_NVIDIA_DRIVER:-1}"
dataset_source="${DATASET_SOURCE:-}"
dataset_username="${DATASET_USERNAME:-}"
dataset_password="${DATASET_PASSWORD:-}"
target_user="${SUDO_USER:-$USER}"
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
setup_tmpdir="${SETUP_TMPDIR:-/var/tmp}"
python_cmd="python3"
python_packages=(python3 python3-pip python3-venv)

if [[ ! -r /etc/os-release ]]; then
    printf 'This script supports Ubuntu LTS only.\n' >&2
    exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    printf 'Expected Ubuntu, found %s %s.\n' "${ID:-unknown}" "${VERSION_ID:-unknown}" >&2
    exit 1
fi
case "${VERSION_ID:-}" in
    22.04|24.04)
        ;;
    26.04)
        # PyTorch 2.5.1 has no CPython 3.14 wheels. The immers.cloud 26.04
        # image ships a deadsnakes repository with a compatible interpreter.
        python_cmd="python3.12"
        python_packages=(python3.12 python3.12-venv)
        ;;
    *)
        printf 'Supported Ubuntu LTS versions are 22.04, 24.04 and 26.04; found %s.\n' \
            "${VERSION_ID:-unknown}" >&2
        exit 1
        ;;
esac

as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

install_system_packages() {
    as_root apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates cron curl file gnupg lsb-release unzip \
        linux-headers-"$(uname -r)" "${python_packages[@]}"

    if [[ "$install_driver" == "1" ]]; then
        driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
            | head -n1 | cut -d. -f1 || true)"
        if [[ -n "$driver_major" && "$driver_major" -ge 535 ]]; then
            printf 'Keeping working NVIDIA driver R%s (R535 or newer is compatible).\n' "$driver_major"
        else
            as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                nvidia-driver-535-server
        fi
    fi
}

install_docker() {
    as_root install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | as_root gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    as_root chmod a+r /etc/apt/keyrings/docker.gpg

    arch="$(dpkg --print-architecture)"
    codename="$(. /etc/os-release && printf '%s' "$VERSION_CODENAME")"
    printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu %s stable\n' \
        "$arch" "$codename" \
        | as_root tee /etc/apt/sources.list.d/docker.list >/dev/null

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | as_root gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        | as_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

    as_root apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
        nvidia-container-toolkit
    as_root nvidia-ctk runtime configure --runtime=docker
    as_root systemctl enable --now docker
    as_root systemctl restart docker
    as_root usermod -aG docker "$target_user"
}

extract_dataset_archive() {
    local archive="$1"
    local temporary_dir="$2"
    local unpacked="${temporary_dir}/unpacked"
    mkdir -p "$unpacked"

    case "$(file -b --mime-type "$archive")" in
        application/zip)
            if [[ -n "$dataset_password" ]]; then
                unzip -q -P "$dataset_password" "$archive" -d "$unpacked"
            else
                unzip -q "$archive" -d "$unpacked"
            fi
            ;;
        application/x-tar|application/gzip|application/x-gzip|application/x-xz)
            tar -xf "$archive" -C "$unpacked"
            nested_zip="$(find "$unpacked" -type f -name '*.zip' -print -quit)"
            if [[ -n "$nested_zip" ]]; then
                nested_dir="${temporary_dir}/nested"
                mkdir -p "$nested_dir"
                if [[ -n "$dataset_password" ]]; then
                    unzip -q -P "$dataset_password" "$nested_zip" -d "$nested_dir"
                else
                    unzip -q "$nested_zip" -d "$nested_dir"
                fi
                unpacked="$nested_dir"
            fi
            ;;
        *)
            printf 'Unsupported dataset archive: %s\n' "$archive" >&2
            return 1
            ;;
    esac

    dataset_dir="$(find "$unpacked" -type f -name train.csv -print -quit | xargs -r dirname)"
    if [[ -z "$dataset_dir" ]]; then
        printf 'train.csv was not found in %s\n' "$archive" >&2
        return 1
    fi
    mkdir -p "$data_root"
    cp -a "${dataset_dir}/." "$data_root/"
}

dataset_is_complete() {
    local required csv_path image_id candidate found missing
    for required in train.csv test_query.csv test_gallery.csv images; do
        if [[ ! -e "${data_root}/${required}" ]]; then
            printf 'Dataset is incomplete: missing %s\n' "${data_root}/${required}" >&2
            return 1
        fi
    done

    for required in train.csv test_query.csv test_gallery.csv; do
        csv_path="${data_root}/${required}"
        missing=""
        while IFS= read -r image_id; do
            image_id="${image_id%$'\r'}"
            found=0
            for candidate in \
                "${data_root}/images/${image_id}" \
                "${data_root}/images/${image_id}.jpg" \
                "${data_root}/images/${image_id}.jpeg" \
                "${data_root}/images/${image_id}.JPG" \
                "${data_root}/images/${image_id}.JPEG"; do
                if [[ -f "$candidate" ]]; then
                    found=1
                    break
                fi
            done
            if [[ "$found" == "0" ]]; then
                missing="$image_id"
                break
            fi
        done < <(awk -F, 'NR > 1 {print $1}' "$csv_path")
        if [[ -n "$missing" ]]; then
            printf 'Dataset is incomplete: image %s from %s is missing.\n' "$missing" "$csv_path" >&2
            return 1
        fi
    done
}

install_dataset() {
    if dataset_is_complete; then
        printf 'Dataset already exists at %s; keeping it.\n' "$data_root"
    else
        if [[ -z "$dataset_source" ]]; then
            printf 'Dataset is absent. Set DATASET_SOURCE to a local directory/archive or an HTTPS URL.\n' >&2
            exit 1
        fi
        # A previous interrupted run may have reached the final read-only step
        # before all files arrived. Make that recoverable and lock it again only
        # after the complete CSV-to-image validation below passes.
        if [[ -e "$data_root" ]]; then
            as_root chmod -R u+w -- "$data_root"
        fi
        mkdir -p "$setup_tmpdir"
        temporary_dir="$(mktemp -d "${setup_tmpdir%/}/falcon-dataset.XXXXXX")"
        trap 'rm -rf -- "${temporary_dir:-}"' EXIT
        if [[ -d "$dataset_source" ]]; then
            mkdir -p "$data_root"
            cp -a "${dataset_source}/." "$data_root/"
        else
            archive="$dataset_source"
            if [[ "$dataset_source" =~ ^https:// ]]; then
                archive="${temporary_dir}/dataset.archive"
                curl_args=(--fail --location --retry 3 --output "$archive")
                if [[ -n "$dataset_username" ]]; then
                    curl_args+=(--user "${dataset_username}:${dataset_password}")
                fi
                curl "${curl_args[@]}" "$dataset_source"
            elif [[ ! -f "$dataset_source" ]]; then
                printf 'DATASET_SOURCE does not exist: %s\n' "$dataset_source" >&2
                exit 1
            fi
            extract_dataset_archive "$archive" "$temporary_dir"
        fi
        rm -rf -- "$temporary_dir"
        trap - EXIT
    fi

    dataset_is_complete || exit 1
    as_root chmod -R a-w -- "$data_root"
}

install_python_environment() {
    "$python_cmd" -m venv "$venv_dir"
    "$venv_dir/bin/python" -m pip install --upgrade 'pip==24.3.1'
    if [[ -f "${repo_dir}/requirements.txt" ]]; then
        "$venv_dir/bin/python" -m pip install \
            --index-url "$pytorch_index_url" \
            'torch==2.5.1' 'torchvision==0.20.1'
        "$venv_dir/bin/python" -m pip install --requirement "${repo_dir}/requirements.txt"
    fi
}

install_watchdog() {
    as_root install -m 0755 "${repo_dir}/scripts/gpu_idle_watchdog.sh" \
        /usr/local/bin/falcon-gpu-idle-watchdog
    printf '*/5 * * * * root /usr/local/bin/falcon-gpu-idle-watchdog\n' \
        | as_root tee /etc/cron.d/falcon-gpu-idle >/dev/null
    as_root chmod 0644 /etc/cron.d/falcon-gpu-idle
    as_root systemctl enable --now cron
}

main() {
    start_seconds="$(date +%s)"
    install_system_packages
    install_docker
    install_dataset
    install_python_environment
    install_watchdog
    elapsed=$(( $(date +%s) - start_seconds ))

    printf 'Server bootstrap finished in %dm%02ds.\n' $((elapsed / 60)) $((elapsed % 60))
    printf 'Log out and in once for docker-group membership to take effect.\n'
    if [[ "$install_driver" == "1" ]]; then
        printf 'Reboot if the R535 kernel module was installed or changed.\n'
    fi
}

main "$@"
