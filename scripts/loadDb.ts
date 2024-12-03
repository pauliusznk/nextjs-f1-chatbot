import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer";
import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "dotenv/config";

type SimilarityMetric = "dot_product" | "cosine" | "euclidean";

const {
  ASTRA_DB_NAMESPACE = "",
  ASTRA_DB_COLLECTION = "",
  ASTRA_DB_API_ENDPOINT = "",
  ASTRA_DB_APPLICATION_TOKEN = "",
  OPEN_API_KEY = "",
} = process.env;

if (
  !ASTRA_DB_NAMESPACE ||
  !ASTRA_DB_COLLECTION ||
  !ASTRA_DB_API_ENDPOINT ||
  !ASTRA_DB_APPLICATION_TOKEN ||
  !OPEN_API_KEY
) {
  console.error("Missing critical environment variables:");
  console.error("ASTRA_DB_NAMESPACE:", !!ASTRA_DB_NAMESPACE);
  console.error("ASTRA_DB_COLLECTION:", !!ASTRA_DB_COLLECTION);
  console.error("ASTRA_DB_API_ENDPOINT:", !!ASTRA_DB_API_ENDPOINT);
  console.error("ASTRA_DB_APPLICATION_TOKEN:", !!ASTRA_DB_APPLICATION_TOKEN);
  console.error("OPEN_API_KEY:", !!OPEN_API_KEY);
  throw new Error("One or more critical environment variables are missing");
}

const openai = new OpenAI({ apiKey: OPEN_API_KEY });

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT as string, {
  namespace: ASTRA_DB_NAMESPACE as string,
});
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});
const createCollection = async (
  similarityMetric: SimilarityMetric = "dot_product"
) => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION as string, {
    vector: { dimension: 1536, metric: similarityMetric },
  });
  console.log(res);
};

const f1Data = [
  "https://en.wikipedia.org/wiki/Formula_One",
  "https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship",
  "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship",
  "https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship",
  "https://www.formula1.com/en/results/2024/races",
];

const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  for await (const url of f1Data) {
    const content = await scrapePage(url);
    const chunks = await splitter.splitText(content);
    for await (const chunk of chunks) {
      const embedding = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: chunk,
        encoding_format: "float",
      });
      const vector = embedding.data[0].embedding;
      const res = await collection.insertOne({
        $vector: vector,
        text: chunk,
      });
      console.log(res);
    }
  }
};

const scrapePage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });
  return (await loader.scrape())?.replace(/<[^>]*>?/gm, "");
};

createCollection().then(() => loadSampleData());
