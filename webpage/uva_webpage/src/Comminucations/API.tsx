import config from '../config/website_config.json';
export async function api_request(content: any) {

    const urlPlusPath = config.WebsiteConfig.DATA_STATUS_API || "http://localhost:5000/question"
    console.log(content)
    if (content === undefined || content === null || content === "") {
        return
    }
    try {
        const response = await fetch(urlPlusPath+ `?data=${content.toString()}`, {
            method: 'GET',
        })
            const data = await response.json();
            console.log("response api_data:", data);
            return data
    } catch (error) {
        console.error('API Error:', error);
        return error
    }
}