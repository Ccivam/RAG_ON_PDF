// doc-pdfjs-gemini.js
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import readline from "readline";

dotenv.config();

const PDFNAME="./Product.pdf"
const COLLECTION_NAME = "langchainjs-product-pdfjs-deepseek-v1";
const BATCH_SIZE = 20;
const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 500;


async function setupVectorStore() {
  console.log("\nChecking if Qdrant collection already exists...");

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

 

  
  const loader = new PDFLoader(PDFNAME, {
    splitPages: true,
  });

  const docs = await loader.load();
  console.log(`Loaded ${docs.length} pages from PDF`);

  
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    separators: ["\n\n", "\n", ". ", " ", ""],
  });

  const chunks = await splitter.splitDocuments(docs);
  console.log(`Created ${chunks.length} chunks`);

  
  let vectorStore = null;
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    console.log(`Processing batch ${i} - ${i + batch.length - 1} (size=${batch.length}) ...`);

    if (!vectorStore) {
      vectorStore = await QdrantVectorStore.fromDocuments(batch, embeddings, {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: COLLECTION_NAME,
      });
    } else {
      await vectorStore.addDocuments(batch);
    }
  }


  return vectorStore;
}

async function queryVectorStore(vectorStore, userQuery, k = 20) {
  const results = await vectorStore.similaritySearch(userQuery, k);
  return results;
}

async function generateAnswerWithGemini(chunks, userQuery) {
 
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  const formattedChunks = chunks.map((d, idx) => {
    const page = d.metadata?.loc?.pageNumber || d.metadata?.page || 'unknown';
    return ` CHUNK ${idx + 1} (Page ${page}) 
${d.pageContent}
`;
  });

  const context = formattedChunks.join("\n\n");

  const prompt = `You are a helpful document assistant. Use ONLY the provided context to answer the question.

Instructions:
- Provide detailed answers from the context
- Look for section numbers (6.7, 8.21, etc.) and their full descriptions
- If information spans multiple chunks, synthesize them
- Cite page numbers used
- If not found, say "Information not found in the document."
- If asked about guidelines of a reference number, do understand that guidelines would be in multiple lines of paragraph after the reference number or before reference number answer that paragraph

Context:
${context}

Question:
${userQuery}

Answer:`;

  try {
    const resp = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    });

   
    const text = 
      resp?.text ||
      resp?.response?.text() ||
      resp?.candidates?.[0]?.content?.parts?.[0]?.text ||
      resp?.content?.[0]?.text ||
      "No response generated";

  
    
    return text;
  } catch (error) {
    console.error("Gemini API error:", error);
    console.error("Full error details:", JSON.stringify(error, null, 2));
    throw error;
  }
}


(async () => {
  try {
    const vectorStore = await setupVectorStore();
    const userQuery ="what is the reference of Digital signatures must be used to safeguard the integrity of GA code";
   
    const relevantDocs = await queryVectorStore(vectorStore, userQuery, 15);

    if (!relevantDocs || relevantDocs.length === 0) {
      console.log("No relevant chunks found.");
      return;
    }
    console.log(relevantDocs)
   
    const answer = await generateAnswerWithGemini(relevantDocs, userQuery);

   
    console.log(answer);
    
  } catch (err) {
    console.error("Fatal error:", err);
    process.exit(1);
  }
})();