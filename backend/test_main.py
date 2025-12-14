import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import os
from main import app

# Create test client
client = TestClient(app)


class TestRootEndpoint:
    """Tests for the root endpoint"""
    
    def test_root_endpoint(self):
        """Test that root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "ApplianceCare AI Assistant API"
        assert data["status"] == "running"
        assert "docs" in data
        assert "redoc" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint"""
    
    def test_health_endpoint(self):
        """Test that health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestQueryEndpoint:
    """Tests for the query endpoint"""
    
    @patch('main.llm')
    @patch('main.index')
    @patch('main.embedding_model')
    def test_query_successful_high_score(self, mock_embedding, mock_index, mock_llm):
        """Test successful query with high relevance score"""
        # Mock embedding
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        # Mock Pinecone query response
        mock_index.query.return_value = {
            'matches': [
                {
                    'score': 0.6,
                    'metadata': {
                        'text': 'Test text about washing machine repair',
                        'source': 'test.pdf',
                        'chunk_index': 1
                    }
                },
                {
                    'score': 0.55,
                    'metadata': {
                        'text': 'More test text about appliances',
                        'source': 'test2.pdf',
                        'chunk_index': 2
                    }
                },
                {
                    'score': 0.5,
                    'metadata': {
                        'text': 'Additional repair information',
                        'source': 'test3.pdf',
                        'chunk_index': 3
                    }
                }
            ]
        }
        
        # Mock OpenAI response
        mock_ai_response = MagicMock()
        mock_ai_response.content = "Based on the context, here's how to fix the issue..."
        mock_llm.invoke.return_value = mock_ai_response
        
        # Mock prompt format_messages
        with patch('main.prompt') as mock_prompt:
            mock_prompt.format_messages.return_value = [MagicMock()]
            
            response = client.post(
                "/query",
                json={"query": "How to fix a washing machine?"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "total_score" in data
        assert data["total_score"] > 25.0  # Should be above threshold
        assert data["answer"] is not None
        assert isinstance(data["total_score"], float)
    
    @patch('main.index')
    @patch('main.embedding_model')
    def test_query_low_score_fallback(self, mock_embedding, mock_index):
        """Test query with low relevance score returns fallback message"""
        # Mock embedding
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        # Mock Pinecone query response with low scores
        mock_index.query.return_value = {
            'matches': [
                {
                    'score': 0.1,
                    'metadata': {
                        'text': 'Unrelated text',
                        'source': 'test.pdf',
                        'chunk_index': 1
                    }
                },
                {
                    'score': 0.12,
                    'metadata': {
                        'text': 'More unrelated text',
                        'source': 'test2.pdf',
                        'chunk_index': 2
                    }
                },
                {
                    'score': 0.08,
                    'metadata': {
                        'text': 'Even more unrelated',
                        'source': 'test3.pdf',
                        'chunk_index': 3
                    }
                }
            ]
        }
        
        response = client.post(
            "/query",
            json={"query": "Completely unrelated question"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "I could not find enough information about this issue in the dataset."
        assert data["total_score"] < 25.0
        assert isinstance(data["total_score"], float)
    
    def test_query_missing_field(self):
        """Test query with missing required field"""
        response = client.post(
            "/query",
            json={}
        )
        assert response.status_code == 422  # Validation error
    
    def test_query_empty_string(self):
        """Test query with empty string"""
        response = client.post(
            "/query",
            json={"query": ""}
        )
        # Should still process but might return empty results
        assert response.status_code in [200, 422]
    
    def test_query_invalid_json(self):
        """Test query with invalid JSON"""
        response = client.post(
            "/query",
            data="invalid json"
        )
        assert response.status_code == 422
    
    @patch('main.index')
    @patch('main.embedding_model')
    def test_query_no_results(self, mock_embedding, mock_index):
        """Test query that returns no results"""
        # Mock embedding
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        # Mock Pinecone query with no matches
        mock_index.query.return_value = {
            'matches': []
        }
        
        response = client.post(
            "/query",
            json={"query": "Very specific question"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_score"] == 0.0
        assert data["answer"] == "I could not find enough information about this issue in the dataset."
    
    @patch('main.llm')
    @patch('main.index')
    @patch('main.embedding_model')
    def test_query_openai_error(self, mock_embedding, mock_index, mock_llm):
        """Test query when OpenAI API fails"""
        # Mock embedding
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        # Mock Pinecone query response with good scores
        mock_index.query.return_value = {
            'matches': [
                {
                    'score': 0.6,
                    'metadata': {
                        'text': 'Test text',
                        'source': 'test.pdf',
                        'chunk_index': 1
                    }
                }
            ]
        }
        
        # Mock OpenAI to raise an error
        mock_llm.invoke.side_effect = Exception("OpenAI API error")
        
        # Mock prompt format_messages
        with patch('main.prompt') as mock_prompt:
            mock_prompt.format_messages.return_value = [MagicMock()]
            
            response = client.post(
                "/query",
                json={"query": "Test question"}
            )
        
        # Should still return 200 but with None answer
        assert response.status_code == 200
        data = response.json()
        assert data["total_score"] > 25.0
        assert data["answer"] is None  # OpenAI failed
    
    @patch('main.index')
    @patch('main.embedding_model')
    def test_query_pinecone_error(self, mock_embedding, mock_index):
        """Test query when Pinecone API fails"""
        # Mock embedding
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        # Mock Pinecone to raise an error
        mock_index.query.side_effect = Exception("Pinecone API error")
        
        response = client.post(
            "/query",
            json={"query": "Test question"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Error querying database" in data["detail"]


class TestResponseModels:
    """Tests for Pydantic models"""
    
    def test_query_request_model(self):
        """Test QueryRequest model validation"""
        from main import QueryRequest
        
        # Valid request
        request = QueryRequest(query="How to fix a washing machine?")
        assert request.query == "How to fix a washing machine?"
        
        # Empty string should be allowed (validation happens at API level)
        request_empty = QueryRequest(query="")
        assert request_empty.query == ""
    
    def test_query_response_model(self):
        """Test QueryResponse model"""
        from main import QueryResponse
        
        # Valid response with answer
        response = QueryResponse(
            answer="Test answer",
            total_score=75.5
        )
        assert response.answer == "Test answer"
        assert response.total_score == 75.5
        
        # Response with None answer
        response_none = QueryResponse(
            answer=None,
            total_score=30.0
        )
        assert response_none.answer is None
        assert response_none.total_score == 30.0


class TestScoreCalculation:
    """Tests for score calculation logic"""
    
    @patch('main.index')
    @patch('main.embedding_model')
    def test_score_calculation_average(self, mock_embedding, mock_index):
        """Test that score is calculated as average percentage"""
        # Mock embedding
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        # Mock Pinecone query with specific scores
        mock_index.query.return_value = {
            'matches': [
                {
                    'score': 0.5,
                    'metadata': {
                        'text': 'Text 1',
                        'source': 'test.pdf',
                        'chunk_index': 1
                    }
                },
                {
                    'score': 0.6,
                    'metadata': {
                        'text': 'Text 2',
                        'source': 'test2.pdf',
                        'chunk_index': 2
                    }
                },
                {
                    'score': 0.4,
                    'metadata': {
                        'text': 'Text 3',
                        'source': 'test3.pdf',
                        'chunk_index': 3
                    }
                }
            ]
        }
        
        with patch('main.prompt') as mock_prompt, patch('main.llm') as mock_llm:
            mock_prompt.format_messages.return_value = [MagicMock()]
            mock_ai_response = MagicMock()
            mock_ai_response.content = "Answer"
            mock_llm.invoke.return_value = mock_ai_response
            
            response = client.post(
                "/query",
                json={"query": "Test"}
            )
        
        assert response.status_code == 200
        data = response.json()
        # Average of 0.5, 0.6, 0.4 = 0.5, so percentage = 50.0
        expected_score = ((0.5 + 0.6 + 0.4) / 3) * 100
        assert abs(data["total_score"] - expected_score) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

