# gguf

[![CI](https://github.com/YdrMaster/gguf/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/YdrMaster/gguf/actions)
[![license](https://img.shields.io/github/license/YdrMaster/gguf)](https://mit-license.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/YdrMaster/gguf)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/YdrMaster/gguf)

[![GitHub Issues](https://img.shields.io/github/issues/YdrMaster/gguf)](https://github.com/YdrMaster/gguf/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/YdrMaster/gguf)](https://github.com/YdrMaster/gguf/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/YdrMaster/gguf)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/YdrMaster/gguf)

## [ggus 库](/ggus)

## gguf 实用工具

### 帮助信息

```plaintext
cargo xtask --help
```

```plaintext
Usage: xtask.exe <COMMAND>

Commands:
  show      Show the contents of gguf files
  split     Split gguf files into shards
  merge     Merge shards into a single gguf file
  filter    Filter gguf files based on wildcard patterns
  convert   Convert gguf files to different format
  set-meta  Set metadata of gguf files
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### 阅读内容

```plaintext
cargo show --help
```

```plaintext
Show the contents of gguf files

Usage: xtask.exe show [OPTIONS] <FILE>

Arguments:
  <FILE>  The file to show

Options:
      --shards                         If set, show all shards in the directory
  -n, --array-detail <ARRAY_DETAIL>    How many elements to show in arrays, `all` for all elements [default: 8]
  -m, --filter-meta <FILTER_META>      Meta to show [default: *]
  -t, --filter-tensor <FILTER_TENSOR>  Tensors to show [default: *]
  -h, --help                           Print help
```

### 内容过滤

```plaintext
cargo filter --help
```

```plaintext
Filter gguf files based on wildcard patterns

Usage: xtask.exe filter [OPTIONS] <FILE>

Arguments:
  <FILE>  The file to filter

Options:
  -o, --output-dir <OUTPUT_DIR>        Output directory for filtered file
  -m, --filter-meta <FILTER_META>      Meta to keep [default: *]
  -t, --filter-tensor <FILTER_TENSOR>  Tensors to keep [default: *]
  -h, --help                           Print help
```

### 分片

```plaintext
cargo split --help
```

```plaintext
Split gguf files into shards

Usage: xtask.exe split [OPTIONS] <FILE>

Arguments:
  <FILE>  File to split

Options:
  -o, --output-dir <OUTPUT_DIR>    Output directory for splited shards
  -t, --max-tensors <MAX_TENSORS>  Max count of tensors per shard
  -s, --max-bytes <MAX_BYTES>      Max size in bytes per shard
  -n, --no-tensor-first            If set, the first shard will not contain any tensor
  -h, --help                       Print help
```

### 合并

```plaintext
cargo merge --help
```

```plaintext
Merge shards into a single gguf file

Usage: xtask.exe merge [OPTIONS] <FILE>

Arguments:
  <FILE>  One of the shards to merge

Options:
  -o, --output-dir <OUTPUT_DIR>  Output directory for merged file
  -h, --help                     Print help
```

### 转换格式

```plaintext
cargo convert --help
```

```plaintext
Convert gguf files to different format

Usage: xtask.exe convert [OPTIONS] --ops <OPS> <FILE>

Arguments:
  <FILE>  File to convert

Options:
  -o, --output-dir <OUTPUT_DIR>    Output directory for converted files
      --ops <OPS>                  Operations to apply, separated by "->"
  -t, --max-tensors <MAX_TENSORS>  Max count of tensors per shard
  -s, --max-bytes <MAX_BYTES>      Max size in bytes per shard
  -n, --no-tensor-first            If set, the first shard will not contain any tensor
  -h, --help                       Print help
```

### 修改元信息

```plaintext
cargo set-meta --help
```

```plaintext
Set metadata of gguf files

Usage: xtask.exe set-meta [OPTIONS] <FILE> <META_KVS>

Arguments:
  <FILE>      File to set metadata
  <META_KVS>  Meta data to set for the file

Options:
  -o, --output-dir <OUTPUT_DIR>  Output directory for changed file
  -h, --help                     Print help
```

`<META_KVS>` 是具有类似如下格式的文本文件：

```plaintext
`general.alignment` u32 128
`tokenizer.chat_template` str|

| {%- for message in messages -%}
| {%- if message['role'] == 'user' -%}
| {{ '<|user|>
| ' + message['content'] + eos_token }}
| {%- elif message['role'] == 'system' -%}
| {{ '<|system|>
| ' + message['content'] + eos_token }}
| {%- elif message['role'] == 'assistant' -%}
| {{ '<|assistant|>
| ' + message['content'] + eos_token }}
| {%- endif -%}
| {%- if loop.last and add_generation_prompt -%}
| {{ '<|assistant|>
| ' }}
| {%- endif -%}
| {%- endfor -%}
```
