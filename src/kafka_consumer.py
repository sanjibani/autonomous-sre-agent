"""
Kafka consumer for real-time log streaming.
Enables streaming ingestion for production-scale log volumes.
"""
import json
import time
import threading
from typing import Callable, List, Optional
from datetime import datetime

# Kafka is optional - graceful fallback if not available
try:
    from confluent_kafka import Consumer, KafkaError, KafkaException
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: confluent-kafka package not installed. Streaming disabled.")

from .config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_LOGS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_BATCH_SIZE,
    KAFKA_BATCH_WINDOW_MS
)


class LogStreamConsumer:
    """
    Kafka consumer that batches incoming logs and passes them to a processor.
    
    Features:
    - Configurable batch size and time window
    - Graceful shutdown
    - Automatic reconnection
    - Thread-safe batch accumulation
    """
    
    def __init__(
        self,
        on_batch: Callable[[List[str], datetime], None],
        bootstrap_servers: str = None,
        topic: str = None,
        group_id: str = None,
        batch_size: int = None,
        batch_window_ms: int = None
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            on_batch: Callback function called with (logs, timestamp) when batch is ready
            bootstrap_servers: Kafka bootstrap servers (default from config)
            topic: Topic to consume from (default from config)
            group_id: Consumer group ID (default from config)
            batch_size: Max logs per batch (default from config)
            batch_window_ms: Max time to wait before flushing batch (default from config)
        """
        if not KAFKA_AVAILABLE:
            raise RuntimeError("confluent-kafka is not installed. Run: pip install confluent-kafka")
        
        self._on_batch = on_batch
        self._bootstrap_servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
        self._topic = topic or KAFKA_TOPIC_LOGS
        self._group_id = group_id or KAFKA_CONSUMER_GROUP
        self._batch_size = batch_size or KAFKA_BATCH_SIZE
        self._batch_window_ms = batch_window_ms or KAFKA_BATCH_WINDOW_MS
        
        self._consumer = None
        self._running = False
        self._thread = None
        self._batch = []
        self._batch_lock = threading.Lock()
        self._last_flush = time.time()
    
    def _create_consumer(self) -> Consumer:
        """Create and configure the Kafka consumer"""
        config = {
            'bootstrap.servers': self._bootstrap_servers,
            'group.id': self._group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 30000,
        }
        return Consumer(config)
    
    def _flush_batch(self):
        """Flush accumulated logs to the processor"""
        with self._batch_lock:
            if not self._batch:
                return
            
            logs = self._batch.copy()
            self._batch = []
            self._last_flush = time.time()
        
        try:
            self._on_batch(logs, datetime.now())
        except Exception as e:
            print(f"Error processing log batch: {e}")
    
    def _should_flush(self) -> bool:
        """Check if batch should be flushed based on size or time"""
        with self._batch_lock:
            if len(self._batch) >= self._batch_size:
                return True
            
            elapsed_ms = (time.time() - self._last_flush) * 1000
            if self._batch and elapsed_ms >= self._batch_window_ms:
                return True
        
        return False
    
    def _consume_loop(self):
        """Main consumption loop (runs in background thread)"""
        self._consumer = self._create_consumer()
        self._consumer.subscribe([self._topic])
        print(f"Kafka consumer started: {self._topic} @ {self._bootstrap_servers}")
        
        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                
                if msg is None:
                    # No message, check if we should flush on timeout
                    if self._should_flush():
                        self._flush_batch()
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"Kafka error: {msg.error()}")
                        continue
                
                # Parse log message
                try:
                    log_data = msg.value().decode('utf-8')
                    
                    # Support both raw text and JSON format
                    try:
                        parsed = json.loads(log_data)
                        log_line = parsed.get('message', parsed.get('log', str(parsed)))
                    except json.JSONDecodeError:
                        log_line = log_data
                    
                    with self._batch_lock:
                        self._batch.append(log_line.strip())
                    
                    if self._should_flush():
                        self._flush_batch()
                
                except Exception as e:
                    print(f"Error parsing Kafka message: {e}")
        
        except KafkaException as e:
            print(f"Kafka exception: {e}")
        
        finally:
            self._consumer.close()
            print("Kafka consumer stopped")
    
    def start(self):
        """Start consuming in a background thread"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
    
    def stop(self, timeout: float = 5.0):
        """Stop the consumer gracefully"""
        if not self._running:
            return
        
        self._running = False
        
        # Flush any remaining logs
        self._flush_batch()
        
        if self._thread:
            self._thread.join(timeout=timeout)
    
    @property
    def is_running(self) -> bool:
        """Check if consumer is active"""
        return self._running and self._thread and self._thread.is_alive()


class LogStreamProducer:
    """
    Simple Kafka producer for testing log streaming.
    """
    
    def __init__(self, bootstrap_servers: str = None, topic: str = None):
        if not KAFKA_AVAILABLE:
            raise RuntimeError("confluent-kafka is not installed")
        
        from confluent_kafka import Producer
        
        self._bootstrap_servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
        self._topic = topic or KAFKA_TOPIC_LOGS
        self._producer = Producer({'bootstrap.servers': self._bootstrap_servers})
    
    def send(self, log: str):
        """Send a single log message"""
        self._producer.produce(self._topic, value=log.encode('utf-8'))
        self._producer.poll(0)
    
    def send_batch(self, logs: List[str]):
        """Send multiple log messages"""
        for log in logs:
            self.send(log)
        self._producer.flush()
    
    def flush(self):
        """Ensure all messages are delivered"""
        self._producer.flush()


def is_kafka_available() -> bool:
    """Check if Kafka is available and configured"""
    return KAFKA_AVAILABLE
