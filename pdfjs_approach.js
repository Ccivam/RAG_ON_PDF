import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const PDFNAME = "./Product.pdf";
const COLLECTION_NAME = "langchainjs-product-pdfjs-deepseek-v1-Newcitation";
const BATCH_SIZE = 20;
const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 500;

async function loadAndSplitPDF() {
  const loader = new PDFLoader(PDFNAME, { splitPages: true });
  const docs = await loader.load();

 
  const docsWithPage = docs.map((doc, idx) => ({
    ...doc,
    metadata: {
      ...doc.metadata,
      source: "Product.pdf",
      page: idx + 1,
    },
  }));

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });

  const chunks = await splitter.splitDocuments(docsWithPage);

  
  return chunks.map((chunk, idx) => ({
    ...chunk,
    metadata: {
      source: "Product.pdf",
      page: chunk.metadata.page, 
      chunk_id: idx + 1,
    },
  }));
}


async function setupDenseVectorStore(chunks) {
  const client = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
  });

  const collections = await client.getCollections();
  const exists = collections.collections.some((c) => c.name === COLLECTION_NAME);

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_API_KEY,
    model: "BAAI/bge-large-en-v1.5",
  });

  if (exists) {
    return await QdrantVectorStore.fromExistingCollection(embeddings, {
      url: process.env.QDRANT_URL,
      apiKey: process.env.QDRANT_API_KEY,
      collectionName: COLLECTION_NAME,
    });
  }
    

  let vectorStore = null;

for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
  const batch = chunks.slice(i, i + BATCH_SIZE).map((doc, idx) => ({
    pageContent: doc.pageContent,
    metadata: {
      source: "Product.pdf",
      page:doc.metadata.page,
      
    },
  }));

  if (!vectorStore) {
    vectorStore = await QdrantVectorStore.fromDocuments(
      batch,
      embeddings,
      {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: COLLECTION_NAME,
      }
    );
  } else {
    await vectorStore.addDocuments(batch);
  }
}


  return vectorStore;
}


async function extractReferenceNumber(query) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  const prompt = `You are an assistant. Determine if the user query is asking about a specific reference number from a document.
If yes, extract the reference number exactly. If no, return "NONE".

Query: "${query}"

Answer:`;

  const resp = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
  });

  const text =
    resp?.text?.trim() ||
    resp?.response?.text()?.trim() ||
    resp?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ||
    resp?.content?.[0]?.text?.trim() ||
    "NONE";

  return text === "NONE" ? null : text;
}

function keywordSearch(chunks, keyword) {
  return chunks.filter((d) => d.pageContent.includes(keyword));
}

async function denseSearch(vectorStore, query, k = 3) {
  return await vectorStore.similaritySearch(query, k);
}

async function generateAnswerWithGemini(chunks, userQuery) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

 const context = chunks
  .map(
    (d) =>
      `CHUNK ${d.metadata.chunk_id}
Source: ${d.metadata.source}
Page: ${d.metadata.page}

${d.pageContent}`
  )
  .join("\n\n");


  const prompt = `
You are a compliance document assistant.

Rules:
- Use ONLY the provided context.
- Every bullet point MUST include a citation.
- Citation format:
  [Source: <file>, Page: <page>]

Context:
${context}

Question:
${userQuery}

Answer (with citations):
`;
const resp = await ai.models.generateContent({ model: "gemini-2.5-flash", contents: prompt, });
  return (
    resp?.text ||
    resp?.response?.text() ||
    resp?.candidates?.[0]?.content?.parts?.[0]?.text ||
    resp?.content?.[0]?.text ||
    "No response generated"
  );
}

(async () => {
  try {
    const allChunks = await loadAndSplitPDF();
    const vectorStore = await setupDenseVectorStore(allChunks);

    const userQuery = "what are the requirements related to supplier security?";

    
    const refNumber = await extractReferenceNumber(userQuery);

    let relevantChunks;
    if (refNumber) {
      
      relevantChunks = keywordSearch(allChunks, refNumber);
    } else {
      relevantChunks = await denseSearch(vectorStore, userQuery, 6);
    }

      

    const answer = await generateAnswerWithGemini(relevantChunks, userQuery);

    
    console.log(answer);
  } catch (err) {
    console.error("Fatal error:", err);
    
  }
})();
