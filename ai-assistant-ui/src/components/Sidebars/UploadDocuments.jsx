import { useState } from "react"
import axios from "axios"

export default function UploadDocuments({business, client}){

  // const [business,setBusiness] = useState("hotel")
  // const [client,setClient] = useState("taj")

  const uploadFile = async (e)=>{

    const file = e.target.files[0]

    if(!file) return

    const formData = new FormData()
    formData.append("file", file)

    try{

      await axios.post(
        `https://llm-rag-document-qa-3.onrender.com/${business}/${client}/upload`,
        formData
      )

      alert("Document uploaded")

    }catch(err){

      console.error(err)
      alert("Upload failed")

    }

  }

  return(

    <div>

      <h3>Upload Documents</h3>

      {/* BUSINESS SELECTOR */}

      <select
        value={business}
        onChange={(e)=>setBusiness(e.target.value)}
      >
        <option value="hotel">Hotel</option>
        <option value="clinic">Clinic</option>
        <option value="restaurant">Restaurant</option>
      </select>

      {/* CLIENT ID INPUT */}

      <input
        type="text"
        placeholder="Client ID (example: taj)"
        value={client}
        onChange={(e)=>setClient(e.target.value)}
      />

      {/* FILE INPUT */}

      <input
        type="file"
        onChange={uploadFile}
      />

    </div>

  )

}
