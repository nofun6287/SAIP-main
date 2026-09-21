"""Evaluate pseudo labels against annotations without exposing annotations to generation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from saip.evaluation import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions','annotations','manifest','out'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    result = evaluate(read(a.predictions),read(a.annotations),[r['id'] for r in read(a.manifest)['videos']])
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
