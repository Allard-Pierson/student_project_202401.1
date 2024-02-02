import React, { useEffect, useState } from "react";
import { api_request } from "../Comminucations/API";

interface Prediction_result {
  Prediction: string;
  Confidence: number;
  Error: string;
}

interface score {
  subject: string;
  score: number;
}

interface Scores {
  [key: string]: score;
}

interface Book {
  Titel_en_ondertitel: string;
  Auteur_UID: string;
  Boekenreeks: string;
  Land_van_publicatie: string;
  Taal_van_het_boek: string;
  geografisch_onderwerp: string;
  LCC_classificatie: string;
  Error: string;
  Uitgever_plaats_publicatiejaar: string;
}

interface ListsectionProps {
  searchQuery: string;
}

function isScores(obj: any): obj is Scores {
  return obj && typeof obj === 'object' && Object.values(obj).every((s: any) => 
    s.hasOwnProperty('subject') && typeof s.subject === 'string' &&
    s.hasOwnProperty('score') && typeof s.score === 'number'
  );
}

// Function to filter and display books
export function Listsection({ searchQuery }: ListsectionProps) {
  const [displayedBooks, setDisplayedBooks] = useState<Book[]>([]);
  const [displayedPrediction, setDisplayedPrediction] = useState<Prediction_result[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchBooks = async () => {
      setLoading(true);
      try {
        const result = await api_request(searchQuery);
        console.log(result);
        if (result && typeof result === 'object') {
          if (result.info && Array.isArray(result.info)) {
            setDisplayedBooks(result.info);
          }
          if (result.score && isScores(result.score)) {
            const predictionResults: Prediction_result[] = Object.values(result.score).map((s: any )=> ({
              Prediction: s.subject,
              Confidence: s.score,
              Error: ""
            }));
            setDisplayedPrediction(predictionResults);
          }
        } else {
          console.error("Error fetching data: Result is undefined or null");
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchBooks();
  }, [searchQuery]);

  return (
    <section>
      {loading && (
        <div className="lds-roller">
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
          <div></div>
        </div>
      )}
      <section id="PredictionList">
        {Array.isArray(displayedPrediction) &&
          displayedPrediction.map((Prediction_result, index) => (
            <section key={index} className="PredictionItem">
              {Prediction_result.Error && <p>{(Prediction_result.Error)}</p>}
              <section className="PredictionInfo">
                <h3>{Prediction_result.Prediction.replace("--",", ")}</h3>
                <p><span>Confidence:</span> {Prediction_result.Confidence}</p>
              </section>
            </section>
          ))}
      </section>
      <section id="bookList">
        {Array.isArray(displayedBooks) &&
          displayedBooks.map((book, index) => (
            <section key={index} className="bookItem">
              {book.Error && <p>{(book.Error)}</p>}
              <section className="bookInfo">
                <h3>{book.Titel_en_ondertitel}</h3>
                <p><span>Auteur:</span> {book.Auteur_UID}</p>
                <p><span>Boekenreeks:</span> {book.Boekenreeks}</p>
                <p><span>Land van publicatie:</span> {book.Land_van_publicatie}</p>
                <p><span>Taal van het boek:</span> {book.Taal_van_het_boek}</p>
                <p><span>Geografisch onderwerp:</span> {book.geografisch_onderwerp}</p>
                <p><span>LCC classificatie:</span> {book.LCC_classificatie}</p>
                <p><span>Uitgever plaats publicatiejaar:</span> {book.Uitgever_plaats_publicatiejaar}</p>
              </section>
            </section>
          ))}
      </section>
    </section>
  );
}
