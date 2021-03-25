// main.cpp

// Unused import for window identification
// #include <windows.h>
// #include <windef.h>
// #include <winuser.h>

// c headers
// #include <Python.h>

// c++ headers
#include <iostream>
#include <list>
#include <algorithm>
#include <string>
#include <utility>
#include <fstream>
#include <vector>
#include <chrono>

// Custom headers
#include <interaction_lib/InteractionLib.h>
#include <interaction_lib/misc/InteractionLibPtr.h>


// Structs and Classes

// Stores gaze events
class GazeEvent
{
  public:
    IL::InteractorId id;
    IL::Timestamp    time;
    bool             gazeGained;

    GazeEvent(const GazeEvent &g1) {id = g1.id; time = g1.time, gazeGained = g1.gazeGained; }

    GazeEvent(IL::InteractorId id, IL::Timestamp time, bool hasFocus) {
        this->id = id;
        this->time = time;
        this->gazeGained = hasFocus;
    }
};

// Stores fixation events
class Fixation
{
  public:
    std::string AOIName;
    long long   startTime;
    float       duration;
    long long   stopTime;
    float       interFixDur;
    float       horzPos;
    float       vertPos;
    int         AOI;

    Fixation(const Fixation &f1) {
        AOIName = f1.AOIName;
        startTime = f1.startTime;
        duration = f1.duration;
        stopTime = f1.stopTime;
        interFixDur = f1.interFixDur;
        horzPos = f1.horzPos;
        vertPos = f1.vertPos;
        AOI = f1.AOI;
    }

    Fixation(std::string AOIName, 
            long long    startTime, 
            float        duration, 
            long long    stopTime, 
            float        interFixDur, 
            float        horzPos,
            float        vertPos,
            int          AOI) 
    {   
        this->AOIName = AOIName;
        this->startTime = startTime;
        this->duration = duration;
        this->stopTime = stopTime;
        this->interFixDur = interFixDur;
        this->horzPos = horzPos;
        this->vertPos = vertPos;
        this->AOI = AOI;
    }
};


// Functions

// Inline convert bools to string
inline const char * const BoolToString(bool b) {
  return b ? "true" : "false";
}

// to_string functionality for GazeEvent class
std::ostream& operator<<(std::ostream &strm, const GazeEvent &a) {
    return strm << "GazeEvent(id: " << a.id << ", time: " << a.time << ", gazeGained: " << BoolToString(a.gazeGained) << ")";
}

// Class to print all data in GazeEvent list
void DumpListData(std::list<GazeEvent> list) {
    while (list.size() > 0)
    {
        GazeEvent ev = list.back();
        std::cout << ev << std::endl;
        // std::cout << list.size() << std::endl;
        list.pop_back();
    }
}

// Calculate coordinates of centre of box depending on id
std::pair<float, float> GetCoordsFromId(int id, int columns, int rows, float width, float height) {
    int colCount = 0;
    int rowCount = 0;
    std::pair<float, float> coords;

    for (size_t i = 0; i < id; i++)
    {
        ++colCount;
        if (colCount == columns)
        {
            ++rowCount;
            colCount = 0;
        }
    }

    // Gets centre of box
    coords.first = (colCount*width) + (width * 0.5);
    coords.second = (rowCount*height) + (height * 0.5);
    return coords;
}

// Main
int main(int argc, char **argv)
{
    // create the interaction library
    IL::UniqueInteractionLibPtr intlib(IL::CreateInteractionLib(IL::FieldOfUse::Interactive));
    
    // setup code vars
    // set numnber of interactor rows and columns
    // assuming they're all the same size
    const int columns = 18;
    const int rows = 7;
    std::cout << "col count: " << columns << ", row count: " << rows << std::endl;
    // setup min fixation time in microseconds
    float minFixLen = 0.1;  // 0.1s
    std::cout << "Minimum fixation length: " << minFixLen << "s" << std::endl;
    // Convert IL::Timestamp to seconds (us -> s)
    float conversion = 1000000;
    // Length of time measurement takes place for in seconds
    constexpr time_t measure_length = 60;
    std::cout << "runtime: " << measure_length << "s" << std::endl;

    // Cannot find window name ~ can't progress to using window based stuff
    // retreive window size of browser
    // HWND hwnd = FindWindowA(NULL, );
    // RECT rect;
    // GetWindowRect(hwnd, &rect);
    // float width = rect.right - rect.left;
    // float height = rect.bottom - rect.top;

    // Set up window area
    const float windowWidth = 1920;
    const float windowHeight = 1080;
    const float offset = 0.0f;  // Coords start in top left area

    intlib->CoordinateTransformAddOrUpdateDisplayArea(windowWidth, windowHeight);
    intlib->CoordinateTransformSetOriginOffset(offset, offset);
    
    const int count = columns * rows;
    const float boxWidth = windowWidth / columns;
    const float boxHeight = windowHeight / rows;

    // Begin setup of interactors
    intlib->BeginInteractorUpdates();

    // setup ids and rectangles that define the interactors we want to use
    int colCount = 0;
    int rowCount = 0;
    const float z = 0.0f;  // This is the depth off the screen (increases towards user)
    for (size_t i = 0; i < count; i++)
    {
        IL::InteractorId id = i;

        // rect = {top_left_point_hori, top_left_point_vert, width, height}
        IL::Rectangle rect = {colCount*boxWidth, rowCount*boxHeight, boxWidth, boxHeight};
        ++colCount;
        if (colCount == columns)
        {
            ++rowCount;
            colCount = 0;
        }
        
        // Add them to the interactor system
        intlib->AddOrUpdateInteractor(id, rect, z);
    }

    // Commit changes
    intlib->CommitInteractorUpdates();

    /* Need by end:
    image name, area of interest on file, fixation start, length and end, gap between fixations, horizontal and vert pos, pupil diam, area of interest

    Next bit is confidence results
    */

    std::list<GazeEvent> gazeVec;

    // subscribe to gaze focus events
    // print event data to std out when called and count the number of consecutive focus events
    intlib->SubscribeGazeFocusEvents([](IL::GazeFocusEvent evt, void* context)
    {
        std::list<GazeEvent>& gazeVec = *static_cast<std::list<GazeEvent>*>(context);

        GazeEvent gaze = GazeEvent(evt.id, evt.timestamp_us, evt.hasFocus);

        gazeVec.push_back(gaze);

    }, &gazeVec);

    // Store start time in ms for use in calcs later
    uint64_t micro_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    // Get start time for running in seconds
    time_t start;
    start = time (NULL);

    // setup and maintain device connection, wait for device data between events and
    // update interaction library to trigger all callbacks
    std::cout << "Starting interaction library update loop.\n";
    IL::Timestamp lastFixTime = 0;

    while (time (NULL) < start + measure_length)
    {
        intlib->WaitAndUpdate();
    }

    std::cout << "Finished monitoring" << std::endl;

    // Setup start time
    GazeEvent gazeTemp = gazeVec.front();
    lastFixTime = gazeTemp.time;
    IL::Timestamp startTime = gazeTemp.time;

    std::list<Fixation> fixations;

    // Get rid of the first entry as it begins incorrectly
    // Seperate out gaze events and write to file
    while (gazeVec.size() > 0)
    {
        GazeEvent gazeStart = gazeVec.front();
        gazeVec.pop_front();

        GazeEvent gazeEnd = gazeVec.front();
        gazeVec.pop_front();

        if (gazeVec.size() == 1)
        {
            break;
        }

        float duration = (gazeEnd.time - gazeStart.time) / conversion;  // Stored in s

        // Check that the start and end are on the same ID and we are entering and leaving the area
        if (!(gazeStart.gazeGained == true && gazeEnd.gazeGained == false && gazeStart.id == gazeEnd.id))
        {
            std::cout << "We have a problem in collecting data\n";
            std::cout << gazeStart << std::endl;
            std::cout << gazeEnd << std::endl;
            gazeVec.push_front(gazeStart);
            // DumpListData(gazeVec);
        }

        else if (duration > minFixLen)
        {
            float interFixTime = (gazeStart.time - lastFixTime) / conversion;
            lastFixTime = gazeEnd.time;
            std::pair<float, float> coords = GetCoordsFromId(gazeStart.id, columns, rows, boxWidth, boxHeight);

            long long fixStartTime = micro_since_epoch + (gazeStart.time - startTime);  // Stored in us
            long long fixEndTime = micro_since_epoch + (gazeEnd.time - startTime);  // Stored in us

            std::string AOINameStart = "Stim";
            Fixation fix = Fixation(AOINameStart + std::to_string(gazeStart.id), fixStartTime, 
                                    duration, fixEndTime, interFixTime, 
                                    coords.first, coords.second, 
                                    gazeStart.id);
                                    
            fixations.push_back(fix);
        }
    }

    // file pointer 
    std::fstream fout;
  
    // opens an existing csv file or creates a new file, add titles
    fout.open("..\\GazemapGen\\Student_data.csv", std::ios::out);
    fout<<"File"<<","<<"AOIName"<<","<<"StartTime"<<","<<"Duration"<<","<<"StopTime"<<","<<"InterfixDur"<<","<<"HorzPos"<<","<<"VertPos"<<","<<"AOI"<<"\n";

    while (fixations.size() > 0)
    {
        Fixation temp = fixations.front();
        fixations.pop_front();
        fout << "test_file" << ", " << temp.AOIName << ", "
            << temp.startTime << ", " << temp.duration << ", "
            << temp.stopTime << ", " << temp.interFixDur << ", "
            << temp.horzPos << ", " << temp.vertPos << ", "
            << temp.AOI << "\n";
    }
    fout.close();

    return(0);
}
