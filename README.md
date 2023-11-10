# questions
The 2nd project of Week 6 of CS50's AI with Python

Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. Among the more famous question answering systems is Watson, the IBM computer that competed (and won) on Jeopardy!. A question answering system of Watson’s accuracy requires enormous complexity and vast amounts of data, but in this problem, very simple question answering system is designed based on inverse document frequency.

This question answering system will perform two tasks: document retrieval and passage retrieval. The system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

This problem uses tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once the most relevant documents are found, there many possible metrics for scoring passages, we use a combination of inverse document frequency and a query term density measure.
