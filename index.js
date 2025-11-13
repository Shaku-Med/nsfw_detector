const express = require('express');
const multer = require('multer');
const { pipeline, RawImage } = require('@huggingface/transformers');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3002;

const upload = multer({
  limits: { fileSize: 100 * 1024 * 1024 },
  storage: multer.memoryStorage()
});

let nsfwPipeline = null;
let initializationAttempts = 0;
const MAX_INIT_ATTEMPTS = 3;

const clearModelCache = () => {
  try {
    const cachePath = path.join(__dirname, 'node_modules', '@huggingface', 'transformers', '.cache', 'AdamCodd', 'vit-base-nsfw-detector');
    if (fs.existsSync(cachePath)) {
      console.log('Clearing corrupted model cache...');
      fs.rmSync(cachePath, { recursive: true, force: true });
      console.log('Cache cleared successfully');
    }
  } catch (error) {
    console.error('Error clearing cache:', error.message);
  }
};

const initializePipeline = async () => {
  if (!nsfwPipeline) {
    try {
      nsfwPipeline = await pipeline('image-classification', 'AdamCodd/vit-base-nsfw-detector');
      initializationAttempts = 0; // Reset on success
      console.log('NSFW detection model loaded successfully');
    } catch (error) {
      initializationAttempts++;
      
      // Check if it's a permission/corruption error (system error 13)
      if (error.message && (error.message.includes('system error number 13') || 
          error.message.includes('failed:system error') ||
          error.message.includes('Permission denied'))) {
        
        console.error(`Model loading failed (attempt ${initializationAttempts}/${MAX_INIT_ATTEMPTS}):`, error.message);
        
        if (initializationAttempts < MAX_INIT_ATTEMPTS) {
          // Clear cache and retry
          clearModelCache();
          console.log('Retrying model initialization...');
          
          // Wait a bit before retrying
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          try {
            nsfwPipeline = await pipeline('image-classification', 'AdamCodd/vit-base-nsfw-detector');
            initializationAttempts = 0;
            console.log('Model loaded successfully after cache clear');
          } catch (retryError) {
            console.error('Retry failed:', retryError.message);
            // Reset pipeline to null so we can try again next time
            nsfwPipeline = null;
            throw retryError;
          }
        } else {
          // Reset pipeline to null so we can try again next time
          nsfwPipeline = null;
          throw new Error('Failed to initialize model after multiple attempts. Please check file permissions and try again.');
        }
      } else {
        // Reset pipeline to null so we can try again next time
        nsfwPipeline = null;
        throw error;
      }
    }
  }
  return nsfwPipeline;
};

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  
  next();
});

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

app.post('/detect/*dynamic', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const pip = await initializePipeline();
    const buffer = req.file.buffer;
    const image = await RawImage.fromBlob(new Blob([buffer], { type: req.file.mimetype || 'image/jpeg' }));
    const result = await pip(image);
    const isNSFW = result[0].label === 'nsfw';

    return res.status(200).json({
      success: true,
      nsfw: isNSFW
    });
  } catch (error) {
    console.error('NSFW detection error:', error);
    return res.status(500).json({
      success: false,
      error: 'Something went wrong!',
      nsfw: false
    });
  }
});

app.use((req, res) => {
  res.status(405).json({ error: 'Method not allowed' });
});

// Initialize pipeline on startup
initializePipeline().catch(error => {
  console.error('Failed to initialize NSFW detection model on startup:', error.message);
  console.log('Model will be initialized on first request');
});

app.listen(PORT, () => {
  console.log(`NSFW detection server running on port ${PORT}`);
});

