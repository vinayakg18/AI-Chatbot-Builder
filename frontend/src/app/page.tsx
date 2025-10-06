"use client";
import { useState } from "react";

type Message = { sender: "user" | "bot"; text: string };

export default function Home() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<Message[]>([]);
  const [filePreview, setFilePreview] = useState("");
  const [isLoading, setIsLoading] = useState(false);

// ----------- Upload File -----------
async function uploadFile(e: React.ChangeEvent<HTMLInputElement>) {
  if (!e.target.files?.length) return;

  const formData = new FormData();
  formData.append("file", e.target.files[0]);
  setIsLoading(true);

  try {
    const res = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);
    } else {
      // ✅ Added summary display
      const summaryText = data.summary
        ? `\n\n📄 Summary:\n${data.summary}`
        : "";

      setFilePreview(
        `File: ${data.filename}\nChunks added: ${data.num_chunks || data.chunks_added}\n\nPreview:\n${data.preview || ""}${summaryText}`
      );
    }
  } catch (err) {
    console.error(err);
    alert("Upload failed");
  }

  setIsLoading(false);
}


  // ----------- Send Chat (use /ask endpoint) -----------
  async function sendMessage() {
    if (!message.trim()) return;
    setChat((prev) => [...prev, { sender: "user", text: message }]);
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message, top_k: 5 }),
      });
      const data = await res.json();

      const answer = data.answer || "No answer received.";
      setChat((prev) => [...prev, { sender: "bot", text: answer }]);
    } catch (err) {
      console.error(err);
      setChat((prev) => [
        ...prev,
        { sender: "bot", text: "Error contacting backend." },
      ]);
    }

    setMessage("");
    setIsLoading(false);
  }

  // ----------- UI Layout -----------
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-3xl font-bold mb-6">AI Chatbot Builder</h1>

      {/* Upload Section */}
      <div className="w-full max-w-md mb-6">
        <label className="block mb-2 text-sm text-gray-300">
          Upload Document
        </label>
        <input
          type="file"
          accept=".pdf,.docx,.csv,.txt"
          onChange={uploadFile}
          className="w-full text-sm text-gray-300 border border-gray-600 rounded-lg cursor-pointer bg-gray-800 focus:outline-none"
        />
        {filePreview && (
          <pre className="mt-3 p-3 bg-gray-800 border border-gray-700 rounded text-xs whitespace-pre-wrap max-h-64 overflow-y-auto">
            {filePreview}
          </pre>
        )}
      </div>

      {/* Chat Section */}
      <div className="flex flex-col w-full max-w-md bg-gray-800 rounded-lg shadow-lg p-4">
        <div className="flex flex-col gap-2 overflow-y-auto h-80 mb-4">
          {chat.map((msg, i) => (
            <div
              key={i}
              className={`p-2 rounded-lg whitespace-pre-wrap ${
                msg.sender === "user"
                  ? "bg-blue-600 self-end text-white"
                  : "bg-gray-700 self-start text-gray-100"
              }`}
            >
              {msg.text}
            </div>
          ))}
          {isLoading && (
            <div className="italic text-gray-400 text-sm self-start">
              Thinking...
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            className="flex-1 p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask about the document..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading}
            className={`px-4 py-2 rounded-md ${
              isLoading
                ? "bg-gray-600 cursor-not-allowed"
                : "bg-blue-500 hover:bg-blue-600 text-white"
            }`}
          >
            Send
          </button>
        </div>
      </div>
    </main>
  );
}
