

// Ask question to assistant
export async function askQuestion(question, business="hotel", client="taj") {

  const response = await fetch(
    `http://localhost:8000/${business}/${client}/query`,
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
    `http://localhost:8000/${business}/${client}/upload`,
    {
      method: "POST",
      body: formData
    }
  )

  return response.json()
}