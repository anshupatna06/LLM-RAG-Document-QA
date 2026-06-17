

// Ask question to assistant
export async function askQuestion(question, business="hotel", client="taj") {

  const response = await fetch(
    `https://llm-rag-document-qa-3.onrender.com/${business}/${client}/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question
      })
    }
  )

  return response.json()
}


// Upload document
export async function uploadFile(file, business="hotel", client="taj") {

  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(
    `https://llm-rag-document-qa-3.onrender.com/${business}/${client}/upload`,
    {
      method: "POST",
      body: formData
    }
  )

  return response.json()
}