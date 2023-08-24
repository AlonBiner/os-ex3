#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <atomic>
#include "MapReduceClient.h"
#include "MapReduceFramework.h"

#define SYSTEM_ERROR_PREFIX "system error: "
/* Error messages */
#define BAD_ALLOC_ERROR_MESSAGE "Bad allocation to memory."
#define FAILURE_TO_CREATE_THREAD_ERROR_MESSAGE "Failed to create thread."
#define PTHREAD_COND_WAIT_ERROR_MESSAGE "error on pthread_cond_wait"
#define PTHREAD_COND_BROADCAST_ERROR "error on pthread_cond_broadcast"
#define PTHREAD_MUTEX_UNLOCK_ERROR_MESSAGE "error on pthread_mutex_unlock"
#define PTHREAD_MUTEX_LOCK_ERROR_MESSAGE "error on pthread_mutex_lock"
#define PTHREAD_MUTEX_DESTROY_ERROR "error on pthread_mutex_destroy"
#define PTHREAD_COND_DESTROY_ERROR_MESSAGE "error on pthread_cond_destroy"

struct ThreadContext;

struct JobContext {
    std::atomic<uint64_t> *jobState;
    pthread_t *threadArray;
    ThreadContext *threadContextArray;
    InputVec inputVector;
    OutputVec &outputVector;
    const MapReduceClient *client;
    std::atomic<uint32_t> *mapPhaseAtomicCounter;
    std::atomic<uint32_t> *intermediatePairsAtomicCounter;
    std::atomic<uint32_t> *shuffleVectorSize;
    pthread_mutex_t shuffleMutex;
    pthread_mutex_t reduceMutex;
    pthread_mutex_t outputMutex;
    pthread_mutex_t waitForJobMutex;
    pthread_cond_t cvThread0;
    pthread_cond_t cvNotThread0;
    int threadsInBarrier;
    int numThreads;
    bool isJobFinished;
    std::vector<IntermediateVec> shufflePhaseVector;
};

struct ThreadContext {
    int id;
    JobContext *jobContext;
    IntermediateVec intermediateVec;
};

/* Function signatures */
void handleError(const std::string &errorMessage);

void initializeJobContext(JobContext **jobContext, int multiThreadLevel, const MapReduceClient *client,
                          const InputVec &inputVec, OutputVec &outputVec);

void *threadEntryPoint(void *arg);

void performMapPhase(void *threadContext, JobContext *jobContext);

void performSortPhase(ThreadContext *threadContext);

void performShufflePhase(ThreadContext *threadContext);

void shuffle(JobContext *jobContext);

void performReducePhase(JobContext *jobContext);

void lockMutex(pthread_mutex_t *mutex);

void unlockMutex(pthread_mutex_t *mutex);

void destroyMutex(pthread_mutex_t *mutex);

void destroyCond(pthread_cond_t *cv);

void condWait(pthread_cond_t *cv, pthread_mutex_t *mutex);

void condBroadcast(pthread_cond_t *cv);

K2 *findMaxK2(JobContext *jobContext, ThreadContext *threadContextArray);

IntermediateVec
popMaxK2sToIntermediateVec(JobContext *jobContext, ThreadContext *threadContextArray, K2 *maxKeyPerIndex);

/**
 * This function starts running the MapReduce algorithm (with several threads) and returns a JobHandle.
 * It is assumed that all the following input arguments are valid.
 * @param client -  The implementation of MapReduceClient or in other words the task that the
 *                  framework should run.
 * @param inputVec – a vector of type std::vector<std::pair<K1*, V1*>>,
 *                   the input elements.
 * @param outputVec – a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
 *                    elements will be added before returning. You can assume that outputVec is empty
 * @param multiThreadLevel – the number of worker threads to be used for running the algorithm.
 *                           You will have to create threads using c function pthread_create. You can assume
 *                           multiThreadLevel argument is valid (greater or equal to 1).
 * @return The function returns JobHandle that will be used for monitoring the job.
 */

JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel) {
    JobContext *jobContext;
    initializeJobContext(&jobContext, multiThreadLevel, &client, inputVec, outputVec);
    return static_cast<JobHandle> (jobContext);
}

/**
 * Initializes the job context's struct.
 * It is assumed that all the following input arguments are valid.
 * @param jobContext - The job context to initialize
 * @param multiThreadLevel - The size of the thread array
 * @param client -  The implementation of MapReduceClient or in other words the task that the
 *                  framework should run.
 * @param inputVec – a vector of type std::vector<std::pair<K1*, V1*>>, the input elements.
 * @param outputVec – a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
 *                    elements will be added before returning. You can assume that outputVec is empty
  */
void initializeJobContext(JobContext **jobContext, int multiThreadLevel, const MapReduceClient *client,
                          const InputVec &inputVec, OutputVec &outputVec) {
    try {
        *jobContext = new JobContext{
                new std::atomic<uint64_t>(0),
                new pthread_t[multiThreadLevel],
                new ThreadContext[multiThreadLevel]{},
                inputVec,
                outputVec,
                client,
                new std::atomic<uint32_t>(0),
                new std::atomic<uint32_t>(0),
                new std::atomic<uint32_t>(0),
                PTHREAD_MUTEX_INITIALIZER,
                PTHREAD_MUTEX_INITIALIZER,
                PTHREAD_MUTEX_INITIALIZER,
                PTHREAD_MUTEX_INITIALIZER,
                PTHREAD_COND_INITIALIZER,
                PTHREAD_COND_INITIALIZER,
                0,
                multiThreadLevel,
                false
        };
    } catch (std::bad_alloc &exception) {
        handleError(BAD_ALLOC_ERROR_MESSAGE);
    }

    for (int threadId = 0; threadId < multiThreadLevel; threadId++) {
        (*jobContext)->threadContextArray[threadId] = ThreadContext{threadId, *jobContext};
        int failureToCreateThread = pthread_create(
                (*jobContext)->threadArray + threadId,
                nullptr,
                threadEntryPoint,
                (*jobContext)->threadContextArray + threadId);
        if (failureToCreateThread) {
            handleError(FAILURE_TO_CREATE_THREAD_ERROR_MESSAGE);
        }
    }
}

/**
 * The function gets an error message, prints it to the stderr and exits with EXIT_FAILURE
 * @param errorMessage - The error message
 */
void handleError(const std::string &errorMessage) {
    std::cout << SYSTEM_ERROR_PREFIX << errorMessage << std::endl;
    exit(1);
}

/**
 * This function is the starting point for the entire Map-Reduce process from the perspective of a single thread.
 * @param arg - A pointer to the threadContext struct of the current thread.
 * @return - nullptr when the process finishes.
 */
void *threadEntryPoint(void *arg) {
    auto threadContext = static_cast<ThreadContext *> (arg);
    JobContext *jobContext = threadContext->jobContext;
    performMapPhase(arg, jobContext);
    performSortPhase(threadContext);
    performShufflePhase(threadContext);
    performReducePhase(jobContext);
    return nullptr;
}

/**
 * Performs the mapping of pairs of the input vector into different threads.
 * @param threadContext - Context of the current running thread
 * @param jobContext - Job context is the context of a job.
 */

void performMapPhase(void *threadContext, JobContext *jobContext) {
    uint32_t currentIndex;

    (*jobContext->jobState) |= ((static_cast<uint64_t>(MAP_STAGE) << 62) |
                                (static_cast<uint64_t>(jobContext->inputVector.size()) << 31));

    while (true) {
        currentIndex = (*jobContext->mapPhaseAtomicCounter)++;

        if (currentIndex >= jobContext->inputVector.size()) {
            break;
        }
        InputPair inputPair = jobContext->inputVector[currentIndex];
        K1 *key1 = inputPair.first;
        V1 *value1 = inputPair.second;
        jobContext->client->map(key1, value1, threadContext);
        (*jobContext->jobState)++;
    }
}

/**
 * Performs the sorting of pairs of the intermediateVec vector of each thread.
 * @param threadContext - Context of the current running thread
 */
void performSortPhase(ThreadContext *threadContext) {
    struct {
        bool operator()(IntermediatePair a, IntermediatePair b) const {
            return *a.first < *b.first;
        }
    } pairComparator;

    std::sort(
            threadContext->intermediateVec.begin(),
            threadContext->intermediateVec.end(),
            pairComparator
    );
}

/**
 * The blocks threads that finished the map and sort phases and unblocks thread 0
 * so it can perform the shuffle operation.
 * Once the shuffle operation finishes, all threads continue running.
 * @param threadContext - The context of the thread which is currently running.
 */
void performShufflePhase(ThreadContext *threadContext) {
    JobContext *jobContext = threadContext->jobContext;

    lockMutex(&jobContext->shuffleMutex);

    if (threadContext->id == 0) {
        if (++(jobContext->threadsInBarrier) < jobContext->numThreads) {
            condWait(&jobContext->cvThread0, &jobContext->shuffleMutex);
        }

        shuffle(jobContext);

        // Initialize reduce phase counter, when there is only one working thread.
        (*jobContext->jobState) = ((static_cast<uint64_t>(REDUCE_STAGE) << 62) |
                                   (static_cast<uint64_t>(jobContext->shuffleVectorSize->load()) << 31));
        jobContext->threadsInBarrier = 0;
        condBroadcast(&jobContext->cvNotThread0);
    } else {
        if (++(jobContext->threadsInBarrier) >= jobContext->numThreads) {
            condBroadcast(&jobContext->cvThread0);
        }
        condWait(&jobContext->cvNotThread0, &jobContext->shuffleMutex);
    }

    unlockMutex(&jobContext->shuffleMutex);
}

/**
 * Executes pthread_cond_broadcast and handles relevant errors.
 * @param cv - The condition variable.
 */
void condBroadcast(pthread_cond_t *cv) {
    if (pthread_cond_broadcast(cv) != 0) {
        handleError(PTHREAD_COND_BROADCAST_ERROR);
    }
}

/**
 * Executes pthread_cond_wait and handles relevant errors.
 * @param cv - The condition variable.
 * @param mutex - The relevant mutex.
 */
void condWait(pthread_cond_t *cv, pthread_mutex_t *mutex) {
    if (pthread_cond_wait(cv, mutex) != 0) {
        handleError(PTHREAD_COND_WAIT_ERROR_MESSAGE);
    }
}

/**
 * Executes pthread_mutex_unlock and handles relevant errors.
 * @param mutex - The mutex.
 */
void unlockMutex(pthread_mutex_t *mutex) {
    if (pthread_mutex_unlock(mutex) != 0) {
        handleError(PTHREAD_MUTEX_UNLOCK_ERROR_MESSAGE);
    }
}

/**
 * Executes pthread_mutex_lock and handles relevant errors.
 * @param mutex - The mutex.
 */
void lockMutex(pthread_mutex_t *mutex) {
    if (pthread_mutex_lock(mutex) != 0) {
        handleError(PTHREAD_MUTEX_LOCK_ERROR_MESSAGE);
    }
}

/**
 * Executes pthread_cond_destroy and handles relevant errors.
 * @param mutex - The condition variable.
 */
void destroyCond(pthread_cond_t *cv) {
    if (pthread_cond_destroy(cv) != 0) {
        handleError(PTHREAD_COND_DESTROY_ERROR_MESSAGE);
    }
}

/**
 * Executes pthread_mutex_destroy and handles relevant errors.
 * @param mutex - The mutex.
 */
void destroyMutex(pthread_mutex_t *mutex) {
    if (pthread_mutex_destroy(mutex) != 0) {
        handleError(PTHREAD_MUTEX_DESTROY_ERROR);
    }
}

/**
 * Performs the actual shuffling (done only by one thread - thread 0) on the intermediateVecs of all threads.
 * @param jobContext - The current job context.
 */
void shuffle(JobContext *jobContext) {
    ThreadContext *threadContextArray = jobContext->threadContextArray;

    (*jobContext->jobState) = ((static_cast<uint64_t>(SHUFFLE_STAGE) << 62) |
                               (static_cast<uint64_t>(jobContext->intermediatePairsAtomicCounter->load()) << 31));

    // find the largest key out of all vectors
    while (true) {
        K2 *maxKeyPerIndex = findMaxK2(jobContext, threadContextArray);

        if (maxKeyPerIndex == nullptr) { // This condition means that all intermediate vectors are empty
            break;
        }
        IntermediateVec intermediateVec = popMaxK2sToIntermediateVec(jobContext, threadContextArray, maxKeyPerIndex);

        jobContext->shufflePhaseVector.push_back(intermediateVec);
        (*jobContext->shuffleVectorSize)++;
    }
}

/**
 * Finds the maximal K2 in the the intermediateVecs of all threads.
 * @param jobContext - Current job context
 * @param threadContextArray - Current job thread contexts array.
 * @return - The maximal K2 value found.
 */
K2 *findMaxK2(JobContext *jobContext, ThreadContext *threadContextArray) {
    K2 *maxKeyPerIndex = nullptr;

    for (int threadId = 0; threadId < jobContext->numThreads; threadId++) {
        IntermediateVec &temp = threadContextArray[threadId].intermediateVec;
        if (temp.empty()) {
            continue;
        } else {
            if (maxKeyPerIndex != nullptr) {
                if (*maxKeyPerIndex < *temp.back().first) {
                    maxKeyPerIndex = temp.back().first;
                }
            } else {
                maxKeyPerIndex = temp.back().first;
            }
        }
    }

    return maxKeyPerIndex;
}

/**
 * pop all max K2s from vectors containing those keys
 * @param jobContext - Job context
 * @param threadContextArray  - Thread context array
 * @param maxKeyPerIndex  - Maximum K2
 * @return An intermediateVec containing all elements with the maximal K2.
 */
IntermediateVec popMaxK2sToIntermediateVec(JobContext *jobContext,
                                           ThreadContext *threadContextArray,
                                           K2 *maxKeyPerIndex) {
    IntermediateVec intermediateVec;

    for (int threadId = 0; threadId < jobContext->numThreads; threadId++) {
        IntermediateVec &temp = threadContextArray[threadId].intermediateVec;

        // pop all max K2s from a vector containing those keys
        while (!temp.empty() &&
               !(*temp.back().first < *maxKeyPerIndex) &&
               !(*maxKeyPerIndex < *temp.back().first)) {
            intermediateVec.push_back(temp.back());
            temp.pop_back();
            (*jobContext->jobState)++;
        }
    }

    return intermediateVec;
}

/**
 * Pops vectors from the shuffle phase vector and performs the reduce operation on them,
 * until the shuffle phase is emptied.
 * @param jobContext - The context of the currently running job.
 */

void performReducePhase(JobContext *jobContext) {
    while (true) {
        lockMutex(&jobContext->reduceMutex);

        if (jobContext->shufflePhaseVector.empty()) {
            unlockMutex(&jobContext->reduceMutex);
            break;
        }

        IntermediateVec intermediateVec;
        intermediateVec = jobContext->shufflePhaseVector.back();
        jobContext->shufflePhaseVector.pop_back();


        unlockMutex(&jobContext->reduceMutex);

        jobContext->client->reduce(&intermediateVec, jobContext);
        (*jobContext->jobState)++;

    }
}


/**
 * The function receives as input intermediary element (K2, V2) and context which contains
 * data structure of the thread that created the intermediary element. The function saves the
 * intermediary element in the context data structures. In addition, the function updates the
 * number of intermediary elements using atomic counter.
 * @param key - The key of the intermediary element
 * @param value - The value of the intermediary element
 * @param context - context which contains data structure of the thread that created the intermediary element.
 */
void emit2(K2 *key, V2 *value, void *context) {
    auto threadContext = static_cast<ThreadContext *> (context);
    JobContext *jobContext = threadContext->jobContext;
    threadContext->intermediateVec.push_back({key, value});
    (*jobContext->intermediatePairsAtomicCounter)++;
}

/**
 * The function receives as input output element (K3, V3) and context which contains data
 * structure of the thread that created the output element. The function saves the output
 * element in the context data structures (output vector). In addition, the function updates the
 * number of output elements using atomic counter.
 * @param key - The key of the output element
 * @param value - The value of the output element
 * @param context - The context of the currently running job.
 */
void emit3(K3 *key, V3 *value, void *context) {
    auto jobContext = static_cast<JobContext *> (context);
    lockMutex(&jobContext->outputMutex);
    jobContext->outputVector.push_back({key, value});
    unlockMutex(&jobContext->outputMutex);
}

/**
 * This function gets JobHandle returned by startMapReduceFramework and waits until it is finished.
 * @param job - the job to wait for.
 */
void waitForJob(JobHandle job) {
    auto jobContext = static_cast<JobContext *> (job);
    lockMutex(&jobContext->waitForJobMutex);

    if (!(jobContext->isJobFinished)) {
        int numOfThreads = jobContext->numThreads;
        for (int i = 0; i < numOfThreads; i++) {
            pthread_join(jobContext->threadArray[i], nullptr);
        }
        jobContext->isJobFinished = true;
    }

    unlockMutex(&jobContext->waitForJobMutex);
}

/**
 * This function gets a JobHandle and updates the state of the job into the given JobState struct.
 * @param job - The job to check it state.
 * @param state - Output arg for the state of the given job.
 */
void getJobState(JobHandle job, JobState *state) {
    auto jobContext = static_cast<JobContext *> (job);
    uint64_t currentJobState = (*jobContext->jobState);
    state->stage = static_cast<stage_t> (currentJobState >> 62 & 3);
    uint32_t completed = (currentJobState & INT32_MAX);
    uint32_t total = (currentJobState >> 31 & INT32_MAX);
    state->percentage = state->stage ? (static_cast<float>(completed) / static_cast<float>(total) * 100) : 0;
}

/**
 * This function releases all resources of a job.
 * @param job - The job to close.
 */
void closeJobHandle(JobHandle job) {
    auto jobContext = static_cast<JobContext *> (job);
    waitForJob(job);

    delete jobContext->jobState;
    delete[] jobContext->threadArray;
    delete[] jobContext->threadContextArray;
    delete jobContext->mapPhaseAtomicCounter;
    delete jobContext->intermediatePairsAtomicCounter;
    delete jobContext->shuffleVectorSize;

    // Destroy mutexes

    destroyMutex(&jobContext->shuffleMutex);
    destroyMutex(&jobContext->reduceMutex);
    destroyMutex(&jobContext->outputMutex);
    destroyMutex(&jobContext->waitForJobMutex);

    // Destroy cond objects

    destroyCond(&jobContext->cvThread0);
    destroyCond(&jobContext->cvNotThread0);

    delete jobContext;
}