'use client'

import { useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState(null)
  const [totalScore, setTotalScore] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setAnswer(null)
    setTotalScore(null)

    try {
      const response = await axios.post(
        'http://localhost:8000/query',
        {
          query: query.trim(),
        }
      )
      setAnswer(response.data.answer)
      setTotalScore(response.data.total_score)
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || 'An error occurred while searching')
      } else {
        setError('An unexpected error occurred')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <div className="header">
        <h1>🔧 ApplianceCare AI Assistant</h1>
        <p>Ask questions about appliance, repair, and maintenance</p>
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <div className="search-input-wrapper">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., How to fix a washing machine that won't drain?"
            className="search-input"
            disabled={loading}
          />
          <button
            type="submit"
            className="search-button"
            disabled={loading || !query.trim()}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {error && (
        <div className="error-message">
          <p>❌ {error}</p>
        </div>
      )}

      {answer && (
        <div className="results-container">
          <div className="answer-header">
            <h2>AI Answer</h2>
            {totalScore !== null && (
              <span className="total-score">
                Confidence: {totalScore.toFixed(1)}%
              </span>
            )}
          </div>
          <div className="answer-card">
            <p className="answer-text">{answer}</p>
          </div>
        </div>
      )}

      <style jsx>{`
        .container {
          max-width: 900px;
          margin: 0 auto;
        }

        .header {
          text-align: center;
          color: white;
          margin-bottom: 3rem;
        }

        .header h1 {
          font-size: 2.5rem;
          margin-bottom: 0.5rem;
          text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }

        .header p {
          font-size: 1.1rem;
          opacity: 0.9;
        }

        .search-form {
          margin-bottom: 2rem;
        }

        .search-input-wrapper {
          display: flex;
          gap: 1rem;
          background: white;
          border-radius: 12px;
          padding: 0.5rem;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .search-input {
          flex: 1;
          border: none;
          outline: none;
          padding: 1rem 1.5rem;
          font-size: 1rem;
          border-radius: 8px;
        }

        .search-button {
          padding: 1rem 2rem;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 8px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .search-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .search-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .error-message {
          background: #fee;
          color: #c33;
          padding: 1rem;
          border-radius: 8px;
          margin-bottom: 2rem;
          border-left: 4px solid #c33;
        }

        .results-container {
          margin-top: 2rem;
        }

        .answer-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .answer-header h2 {
          color: white;
          font-size: 1.5rem;
          margin: 0;
        }

        .total-score {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-size: 0.9rem;
          font-weight: 600;
        }

        .answer-card {
          background: white;
          border-radius: 12px;
          padding: 2rem;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .answer-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }

        .answer-text {
          color: #333;
          line-height: 1.8;
          font-size: 1.1rem;
          white-space: pre-wrap;
        }

        @media (max-width: 768px) {
          .header h1 {
            font-size: 2rem;
          }

          .search-input-wrapper {
            flex-direction: column;
          }

          .search-button {
            width: 100%;
          }

          .answer-header {
            flex-direction: column;
            align-items: flex-start;
          }
        }
      `}</style>
    </main>
  )
}

