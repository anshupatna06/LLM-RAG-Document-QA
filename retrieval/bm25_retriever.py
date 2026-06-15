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

        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, k=3):

        if not self.bm25:
            return []

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        try:
            k = int(k)
        except:
            k = 5

        for idx, score in ranked[:k]:

            results.append((score, self.chunks[idx]["text"], self.chunks[idx]))

        return results