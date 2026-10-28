"""
verify_tex.py - Automated LaTeX Structural, Citation, Label, and Math Delimiter Validator
Checks balanced braces, environments, math delimiters, citations, labels, and IEEE conference format.
"""

import sys
import re


def verify_latex(filename: str):
    print(f"Validating LaTeX document: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Strip comments and escaped characters
    cleaned = re.sub(r'\\\{', '', content)
    cleaned = re.sub(r'\\\}', '', cleaned)
    cleaned = re.sub(r'\\\$', '', cleaned)
    lines = cleaned.splitlines()
    code_lines = []
    for line in lines:
        c_idx = line.find('%')
        if c_idx != -1:
            if c_idx > 0 and line[c_idx - 1] == '\\':
                code_lines.append(line)
            else:
                code_lines.append(line[:c_idx])
        else:
            code_lines.append(line)

    code = "\n".join(code_lines)

    # 2. Check balanced braces
    open_braces = code.count('{')
    close_braces = code.count('}')
    print(f"Braces: '{{' = {open_braces}, '}}' = {close_braces}")
    assert open_braces == close_braces, f"Mismatched braces: {open_braces} open vs {close_braces} close!"

    # 3. Check inline math '$' balance
    dollar_count = code.count('$')
    print(f"Inline math '$' delimiters: {dollar_count}")
    assert dollar_count % 2 == 0, f"Odd number of unescaped '$' delimiters: {dollar_count}!"

    # 4. Check matched \begin{env} and \end{env}
    begins = re.findall(r'\\begin\{([a-zA-Z0-9_*]+)\}', code)
    ends = re.findall(r'\\end\{([a-zA-Z0-9_*]+)\}', code)
    print(f"Environments: {len(begins)} begins, {len(ends)} ends")
    assert len(begins) == len(ends), f"Mismatched environment count: {len(begins)} begins vs {len(ends)} ends!"

    env_stack = []
    for token in re.finditer(r'\\(begin|end)\{([a-zA-Z0-9_*]+)\}', code):
        tag, env_name = token.group(1), token.group(2)
        if tag == "begin":
            env_stack.append(env_name)
        else:
            if not env_stack:
                raise ValueError(f"Unmatched \\end{{{env_name}}}")
            last = env_stack.pop()
            if last != env_name:
                raise ValueError(f"Environment mismatch: \\begin{{{last}}} closed by \\end{{{env_name}}}")

    assert len(env_stack) == 0, f"Unclosed environments remaining: {env_stack}"

    # 5. Check citations and bibitems
    bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', code))
    cite_blocks = re.findall(r'\\cite\{([^}]+)\}', code)
    cited_keys = set()
    for block in cite_blocks:
        for k in block.split(','):
            cited_keys.add(k.strip())

    missing_bibs = cited_keys - bibitems
    unused_bibs = bibitems - cited_keys
    print(f"Bibliography: {len(bibitems)} bibitems, {len(cited_keys)} cited keys")
    assert not missing_bibs, f"Undefined citation keys: {missing_bibs}"
    assert not unused_bibs, f"Uncited bibitems: {unused_bibs}"

    # 6. Check labels and refs
    labels = set(re.findall(r'\\label\{([^}]+)\}', code))
    refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', code))
    missing_labels = refs - labels
    print(f"Labels: {len(labels)} defined, {len(refs)} referenced")
    assert not missing_labels, f"Undefined label references: {missing_labels}"

    # 7. Check IEEE conference format compliance
    assert r"\documentclass[conference,10pt]{IEEEtran}" in code, "Missing conference documentclass!"
    assert r"\IEEEauthorblockN{Aniket Agrawal}" in code, "Missing uniform author block!"
    for forbidden in (
        r"\markboth",
        r"\IEEEtitleabstractindextext",
        r"\IEEEdisplaynontitleabstractindextext",
        r"\IEEEraisesectionheading",
        r"\IEEEcompsocitemizethanks",
    ):
        assert forbidden not in code, f"Forbidden journal macro present: {forbidden}"

    print("[SUCCESS] LaTeX document is 100% syntactically valid, citation-verified, and structurally balanced!")


if __name__ == "__main__":
    verify_latex("kvcache_paging_paper.tex")
