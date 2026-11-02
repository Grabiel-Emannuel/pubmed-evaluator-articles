#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Uso: $0 <arquivo_entrada.docx> <arquivo_saida.pdf>" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

input_file=$1
output_file=$2

if [[ ! -f "$input_file" ]]; then
    echo "Arquivo de entrada não encontrado: $input_file" >&2
    exit 1
fi

if [[ "${input_file,,}" != *.docx ]]; then
    echo "O arquivo de entrada precisa ter extensão .docx." >&2
    exit 1
fi

if [[ "${output_file,,}" != *.pdf ]]; then
    echo "O arquivo de saída precisa ter extensão .pdf." >&2
    exit 1
fi

if ! command -v flatpak >/dev/null 2>&1; then
    echo "Flatpak não está instalado. Instale-o e o LibreOffice antes de converter." >&2
    exit 1
fi

output_dir=$(dirname -- "$output_file")
mkdir -p -- "$output_dir"
input_dir=$(cd -- "$(dirname -- "$input_file")" && pwd -P)
input_name=$(basename -- "$input_file")
input_file="$input_dir/$input_name"
output_dir=$(cd -- "$output_dir" && pwd -P)
output_file="$output_dir/$(basename -- "$output_file")"

temp_dir=$(mktemp -d "$output_dir/.docx-to-pdf.XXXXXX")
trap 'rm -rf -- "$temp_dir"' EXIT

input_stem=${input_name%.*}
generated_pdf="$temp_dir/$input_stem.pdf"

filesystem_args=(--filesystem="$output_dir")
if [[ "$input_dir" != "$output_dir" ]]; then
    filesystem_args+=(--filesystem="$input_dir:ro")
fi

flatpak run "${filesystem_args[@]}" org.libreoffice.LibreOffice \
    --headless --convert-to pdf --outdir "$temp_dir" "$input_file"

if [[ ! -f "$generated_pdf" ]]; then
    echo "O LibreOffice não gerou o PDF esperado: $generated_pdf" >&2
    exit 1
fi

mv -f -- "$generated_pdf" "$output_file"
echo "PDF gerado em: $output_file"
