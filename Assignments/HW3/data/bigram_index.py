import os
import sys
import re
from mrjob.job import MRJob
from mrjob.step import MRStep

class BiIndexJob(MRJob):
    TARGETS = {
        "computer science",
        "information retrieval",
        "power politics",
        "los angeles",
        "bruce willis"
    }

    def steps(self):
        return [
            MRStep(mapper=self.find_bigrams, reducer=self.doc_counter),
            MRStep(reducer=self.alpha_sort)
        ]

    def find_bigrams(self, _, line):
        parts = line.split('\t', 1)
        if len(parts) < 2:
            return
        doc = parts[0]
        text = re.sub(r'[^a-zA-Z\s]', ' ', parts[1]).lower()
        tokens = [w.strip() for w in text.split() if w.strip()]
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in self.TARGETS:
                yield bigram, doc

    def doc_counter(self, bigram, docs):
        freq = {}
        for doc in docs:
            freq[doc] = freq.get(doc, 0) + 1
        sorted_docs = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        output = ' '.join(f"{doc}:{cnt}" for doc, cnt in sorted_docs)
        yield None, (bigram, output)

    def alpha_sort(self, _, results):
        for bigram, line in sorted(results):
            yield bigram, line

if __name__ == "__main__":
    if len(sys.argv) == 1:
        folder = "devdata"
        if not os.path.isdir(folder):
            print("Missing 'devdata' folder.")
            sys.exit(1)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        sys.argv.extend(files)

    job = BiIndexJob()
    runner = job.make_runner()
    with runner:
        runner.run()
        with open("selected_bigram_index.txt", "w") as output_file:
            for line in runner.cat_output():
                output_file.write(line.decode("utf-8"))
