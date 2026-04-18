// // import axios from "axios"

// // const API_URL = "http://localhost:8000"

// // export const askQuestion = async (question) => {

// //   const response = await axios.post(
// //     `${API_URL}/demo_hotel/query`,
// //     {
// //       question: question,
// //       top_k: 3,
// //       threshold: 0.2
// //     }
// //   )

// //   return response.data
// // }
// // export async function askQuestion(question, business="hotel") {

// //   const response = await fetch(`http://localhost:8000/${business}/query`, {
// //     method: "POST",
// //     headers: {
// //       "Content-Type": "application/json"
// //     },
// //     body: JSON.stringify({
// //       question: question,
// //       threshold: 0.30,
// //       top_k: 3
// //     })
// //   })

// //   return response.json()
// // }

// export async function uploadFile(file, business="hotel") {

//   const formData = new FormData()
//   formData.append("file", file)

//   const response = await fetch(
//     `http://localhost:8000/${business}/upload`,
//     {
//       method: "POST",
//       body: formData
//     }
//   )

//   return response.json()
// }

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