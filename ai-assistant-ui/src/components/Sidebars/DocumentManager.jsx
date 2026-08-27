import { useEffect, useState } from "react"
import axios from "axios"

export default function DocumentManager(){

  const [docs,setDocs] = useState([])

  useEffect(()=>{

    axios.get("http://localhost:8000")
      .then(res=>{

        const data = res.data || {}

        // flatten documents from all businesses safely
        const allDocs = Object.entries(data).flatMap(
          ([business, files]) =>
            (files || []).map(f => ({
              name: f,
              business: business
            }))
        )

        setDocs(allDocs)

      })
      .catch(()=>setDocs([]))

  },[])

  const deleteDoc = async(doc)=>{

    await axios.delete(
      "http://localhost:8000",
      {data:{filename: doc.name, business: doc.business}}
    )

    setDocs(docs.filter(d=>d.name!==doc.name))

  }

  return(

    <div>

      <h3>Documents</h3>

      {docs.length === 0 && <div>No documents uploaded</div>}

      {docs.map((d,i)=>(
        <div key={i}>

          <b>{d.business}</b> : {d.name}

          <button onClick={()=>deleteDoc(d)}>
            delete
          </button>

        </div>
      ))}

    </div>

  )

}