import { Dialog } from "@headlessui/react";
import { createContext, useState } from "react";
import { ReactNode } from "react";

export const ModalContext = createContext<
  | {
      openModal: (_: JSX.Element) => void;
      closeModal: () => void;
    }
  | undefined
>(undefined);

export function ModalProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [component, setComponent] = useState(<></>);

  return (
    <ModalContext.Provider
      value={{
        openModal: (x: JSX.Element) => {
          setComponent(x);
          setIsOpen(true);
        },
        closeModal: () => {
          setIsOpen(false);
          setComponent(<></>);
        },
      }}
    >
      {children}
      <Dialog as="div" open={isOpen} onClose={() => setIsOpen(false)}>
        <span className="fixed inset-0 bg-black/90" />
        <div className="fixed inset-0 flex items-center justify-center">
          <Dialog.Panel>{component}</Dialog.Panel>
        </div>
      </Dialog>
    </ModalContext.Provider>
  );
}
