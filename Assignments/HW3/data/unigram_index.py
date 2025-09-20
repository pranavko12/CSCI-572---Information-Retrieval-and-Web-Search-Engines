import os
import sys
import re
from mrjob.job import MRJob
from mrjob.step import MRStep

class UniIndexJob(MRJob):
    MRJob.RUNNER = 'inline'
    def steps(self):
        return [
            MRStep(mapper=self.tokenize_text, reducer=self.combine_counts),
            MRStep(reducer=self.sort_output)
        ]

    def tokenize_text(self, _, line):
        parts = line.split('\t', 1)
        if len(parts) < 2:
            return
        doc = parts[0]
        raw = parts[1]
        clean = re.sub(r'[^a-zA-Z\s]', ' ', raw).lower()
        counts = {}
        for word in clean.split():
            word = word.strip()
            if word:
                counts[word] = counts.get(word, 0) + 1
        for word, count in counts.items():
            yield word, (doc, count)

    def combine_counts(self, word, docs):
        sorted_docs = sorted(docs, key=lambda x: (-x[1], x[0]))
        line = ' '.join(f"{d}:{c}" for d, c in sorted_docs)
        yield None, (word, line)

    def sort_output(self, _, word_lines):
        for word, result in sorted(word_lines):
            yield word, result

if __name__ == "__main__":
    if len(sys.argv) == 1:
        folder = "fulldata"
        if not os.path.isdir(folder):
            print("Missing 'fulldata' folder.")
            sys.exit(1)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        sys.argv.extend(files)

    job = UniIndexJob()
    runner = job.make_runner()
    with runner:
        runner.run()
        with open("unigram_index.txt", "w") as output_file:
            for line in runner.cat_output():
                output_file.write(line.decode("utf-8"))
