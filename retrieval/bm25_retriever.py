from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, chunks):
    

        if not chunks:
            self.bm25=None
            self.tokenized = []
            self.chunks = []
            return
        
        self.chunks = chunks
        
        self.texts = [c["text"] for c in chunks]
        self.tokenized = [t.lower().split() for t in self.texts]

        print("\n========== BM25 INPUT ==========")

        for i, chunk in enumerate(chunks):
            print(i, repr(chunk["text"]))

        print("TOTAL BM25 INPUT:", len(chunks))

        self.bm25 = BM25Okapi(self.tokenized)

        print("\n========== BM25 INDEX ==========")

        for i, text in enumerate(self.texts):
            print(i, repr(text))

        print("TOTAL BM25 INDEX:", len(self.texts))

    def search(self, query, k=3):

        if not self.bm25:
            return []

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        print("\n" + "=" * 80)
        print("🔬 FULL RAW BM25 SCORE INSPECTION")
        print("=" * 80)

        print("BM25 QUERY:", repr(query))

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        for rank, (idx, score) in enumerate(ranked, start=1):

            text = self.chunks[idx]["text"]

            if (
                "laundry" in text.lower()
                or "room service" in text.lower()
            ):
                print(
                    f"RANK={rank} | "
                    f"INDEX={idx} | "
                    f"SCORE={float(score):.6f} | "
                    f"TEXT={text}"
                )

        # ranked = sorted(
        #     list(enumerate(scores)),
        #     key=lambda x: x[1],
        #     reverse=True
        # )

        results = []

        try:
            k = int(k)
        except:
            k = 5

        for idx, score in ranked[:k]:

            #results.append((score, self.chunks[idx]["text"], self.chunks[idx]))
            results.append({

                "score": score,

                "text": self.chunks[idx]["text"],

                "source": self.chunks[idx]

            })
        # print("="* 100)
        # print("========== BM25 INDEX ==========")

        # for i, text in enumerate(self.texts[:30]):
        #     print(i, text)

        # for i, text in enumerate(self.texts):

        #     if "laundry" in text.lower():
        #         print(i, text)


        # for idx, score in enumerate(scores):

        #     text = self.chunks[idx]["text"]

        #     if "laundry" in text.lower():
        #         print(
        #             "Laundry score:",
        #             score,
        #             text
        #         )

        # for idx, score in enumerate(scores):

        #     text = self.chunks[idx]["text"]

        #     if "room service" in text.lower():
        #         print(
        #         "room service score:",
        #         score,
        #         text
        #     )

        # print("="* 100)


        print("\n" + "=" * 80)
        print("========== BM25 INDEX ==========")
        print("=" * 80)

        for i, text in enumerate(self.texts):

            if "laundry" in text.lower() or "laundry service" in text.lower():
                print(i, text)

        return results