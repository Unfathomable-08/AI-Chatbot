const chatApp = () => {
    return {
        input: '',
        loading: false,
        messages: [],

        sentMessage() {
            if (this.input.trim() !== "") {
                console.log(this.input)
                this.loading = true;
                this.messages.push({ sender: 'you', text: this.input, _id: this.messages.length + 1 });
                console.log(this.messages)

                fetchFn(this.input).then(reply => {
                    this.input = '';
                    console.log(reply);
                    this.messages.push({ sender: 'bot', text: reply, _id: this.messages.length + 1 });
                    this.loading = false;
                });
            }
        }
    }
}

const fetchFn = async (msg) => {
    // Send to Flask
    try {
        const res = await fetch('/send-message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: msg })
        });

        const data = await res.json();

        return data.reply

    } catch (error) {
        console.error('Error sending message:', error);
    }
}