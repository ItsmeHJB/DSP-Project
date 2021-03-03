// main.cpp

// Unused import for window identification
// #include <windows.h>
// #include <windef.h>
// #include <winuser.h>

// c++ headers
#include <iostream>
#include <list>
#include <algorithm>
#include <string>
#include <utility>

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

    coords.first = colCount*width;
    coords.second = rowCount*height;
    return coords;
}

// Main
int main()
{
    
    // create the interaction library
    IL::UniqueInteractionLibPtr intlib(IL::CreateInteractionLib(IL::FieldOfUse::Interactive));
    
    // setup code vars
    // set numnber of interactor rows and columns
    // assuming they're all the same size
    const int columns = 3;
    const int rows = 2;
    // setup min fixation time in microseconds
    IL::Timestamp minFixLen = 100000;  // 0.1s

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
        // std::cout
        //     << "Interactor: " << evt.id
        //     << ", focused: " << std::boolalpha << evt.hasFocus
        //     << ", timestamp: " << evt.timestamp_us << " us"
        //     << "\n";

        GazeEvent gaze = GazeEvent(evt.id, evt.timestamp_us, evt.hasFocus);

        gazeVec.push_back(gaze);

    }, &gazeVec);

    // setup and maintain device connection, wait for device data between events and
    // update interaction library to trigger all callbacks
    std::cout << "Starting interaction library update loop.\n";
    constexpr size_t max_focus_count = 10;

    IL::Timestamp lastFixTime = 0;
    while (gazeVec.size() < max_focus_count)
    {
        intlib->WaitAndUpdate();
    }

    // Get rid of the first entry as it begins incorrectly
    // Seperate out gaze events and write to file
    while (gazeVec.size() > 0)
    {
        DumpListData(gazeVec);

        GazeEvent gazeStart = gazeVec.front();
        gazeVec.pop_front();

        GazeEvent gazeEnd = gazeVec.front();
        gazeVec.pop_front();

        // std::cout
        //     << "Interactor: " << gazeStart.id
        //     << ", focused: " << std::boolalpha << gazeStart.gazeGained
        //     << ", timestamp: " << gazeStart.time << " us"
        //     << "\n";

        //image name, area of interest on file, fixation start, length and end, gap between fixations, horizontal and vert pos, pupil diam, area of interest

        // Check that the start and end are on the same ID and we are entering and leaving the area
        if (!(gazeStart.gazeGained == true && gazeEnd.gazeGained == false && gazeStart.id == gazeEnd.id))
        {
            std::cout << "We have a problem in collecting data\n";
            std::cout << gazeStart << std::endl;
            std::cout << gazeEnd << std::endl;
            gazeVec.push_front(gazeStart);
            // DumpListData(gazeVec);

            continue;
        }

        IL::Timestamp duration = gazeEnd.time - gazeStart.time;
        if (duration > minFixLen)
        {
            IL::Timestamp interFixTime = gazeStart.time - lastFixTime;
            lastFixTime = gazeEnd.time;
            std::pair<float, float> coords = GetCoordsFromId(gazeStart.id, columns, rows, boxWidth, boxHeight);

            std::cout << gazeStart.id << ", " << gazeStart.time << ", " << duration << ", " << gazeEnd.time << ", " << interFixTime << ", " << coords.first << ", " << coords.second << std::endl;
        }
    }
    
    system("pause");
    return(0);
}
